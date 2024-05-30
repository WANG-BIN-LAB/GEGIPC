import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def lmax_L(L):
    """Compute largest Laplacian eigenvalue"""
    return np.round(sp.linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False, ncv=100)[0],3)

def rescale_L(L, lmax=2):
    """Rescale Laplacian eigenvalues to [-1,1]"""
    M, M = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype) #
    L /= lmax * 2
    L -= I
    return L

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)#.requires_grad_()

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_selectedGene):
        super().__init__()
        self.name = 'ae'
        self.state = 'train'  # or 'test'
        self.inputdim = input_dim
        self.hidden = 32
        self.outputdim = output_dim

        self.CL1_K = 3
        self.CL1_F = 3
        self.poolsize = 8
        self.FC1Fin = self.CL1_F * (num_selectedGene // self.poolsize)
        self.cl1 = nn.Linear(self.CL1_K, self.CL1_F)
        self.fc1 = nn.Linear(self.FC1Fin, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.dropout = nn.Dropout()

        # tape's encoder
        self.encoder = nn.Sequential(nn.Dropout(),
                                     nn.Linear(self.inputdim, 512),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 256),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     nn.Linear(64, 32),
                                     )


        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))

        self.sum = nn.Linear(32, self.outputdim)

    def graph_conv_cheby(self, x, cl, L, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size();
        B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)#max eigenvalue
        L = rescale_L(L, lmax)

        # convert scipy sparse matric L to pytorch
        L = sparse_mx_to_torch_sparse_tensor(L)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin * B])  # V x Fin*B
        x = x0.unsqueeze(0)  # 1 x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm().forward(L, x0)  # V x Fin*B  #make multiplication
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm().forward(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B --> K x V x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])  # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])  # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)  # B*V x Fout
        x = x.view([B, V, Fout])  # B x V x Fout

        return x

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V ##permute:transform data dim
            x = nn.MaxPool1d(p)(x)  # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self, x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x / x_sum

    def sigmatrix(self):
        w0 = self.decoder[0].weight.T
        w1 = self.decoder[1].weight.T
        w2 = self.decoder[2].weight.T
        w3 = self.decoder[3].weight.T
        w4 = self.decoder[4].weight.T
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = F.hardtanh(torch.mm(w03, w4), 0, 1)
        return w04

    def forward(self, x_gcn, x_nn, L):
        #encoder
        x_ = x_gcn.unsqueeze(2)
        x_ = self.graph_conv_cheby(x_, self.cl1, L[0], self.CL1_F, self.CL1_K)
        x_ = F.relu(x_)
        x_ = self.graph_max_pool(x_, self.poolsize)
        x_ = x_.view(-1, self.FC1Fin)
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        x_ = self.dropout(x_)
        x_ = self.fc2(x_)
        x_ = F.relu(x_)
        x_ = self.dropout(x_)
        x_gcn = self.fc3(x_)

        sigmatrix = self.sigmatrix()
        x_nn = self.encode(x_nn)

        #x_sum = torch.cat((x_gcn, x_nn), 1)
        x_sum = 0.5 *x_nn+0.5 *x_gcn
        z = self.sum(x_sum)
        z = F.softmax(z, dim=1)
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix
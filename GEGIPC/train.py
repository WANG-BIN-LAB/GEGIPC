import copy
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from evaluation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

from model import AutoEncoder

class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * pow(0.9 , epoch)
    if(lr<=1e-5):
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def training_stage(model, train_loader, test_x, test_y, optimizer, geneind, L, epochs=10, seed=0):
    loss = []
    recon_loss = []
    best = -1
    best_model = ''

    for i in range(epochs):
        model.train()
        #adjust_learning_rate(optimizer, i, 1e-3)
        for k, (data, label) in enumerate(train_loader):
            x = data[:,geneind]
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(x, data, L)
            batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data)
            #print('loss:',F.l1_loss(cell_prop, label).item(),';re loss',F.l1_loss(x_recon, data).item())
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

        #test
        model.eval()
        data = torch.from_numpy(test_x).float().to(device)
        data_ = data[:, geneind]
        _, pred, _ = model(data_, data, L)
        pred = pred.cpu().detach().numpy()
        a, b, c, d = score(pred, test_y)
        if c>best:
            best=c
            best_model = copy.deepcopy(model)
        print('===this is the %d epoch of %d epochs==='%(i, epochs))
        print('mae:', np.round(a, 3), 'rmse:', np.round(b, 3), 'ccc:', np.round(c, 3), 'ccc_mean:', np.round(d, 3))
    # if best_model != '':
    #     torch.save(best_model, 'model/model_'+str(seed)+ '.pt')

    return best_model, model, loss, recon_loss

def train_model(train_x, train_y, test_x, test_y, geneind, L,
                batch_size=128, iteration=5000, seed=0):
    reproducibility(seed)
    train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)

    model = AutoEncoder(train_x.shape[1], train_y.shape[1], len(geneind)).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    # best_model, model, loss, reconloss = training_stage(model, train_loader, test_x, test_y, optimizer, geneind, L,
    #                                         epochs=int(iteration / (len(train_x) / 128)), seed = seed)
    best_model, model, loss, reconloss = training_stage(model, train_loader, test_x, test_y, optimizer, geneind, L,
                                            epochs=20 ,seed = seed)
    #print('prediction loss is:')
    #showloss(loss)
    #print('reconstruction loss is:')
    #showloss(reconloss)
    return best_model, model
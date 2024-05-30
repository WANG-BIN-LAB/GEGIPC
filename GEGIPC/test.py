import anndata
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from evaluation import *
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

def laplacian(W, normalized=True):
    """Return graph Laplacian"""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

#    assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sp.csr.csr_matrix
    return L

def adjMatrix(scRNA):
    # temp = anndata.AnnData(X=scRNA.to_numpy(),dtype=scRNA.to_numpy().dtype, var=pd.DataFrame(index=scRNA.columns),
    #                         obs=pd.DataFrame(index=scRNA.index))
    temp = anndata.AnnData(X=scRNA.to_numpy(),dtype=scRNA.to_numpy().dtype, var=pd.DataFrame(index=scRNA.columns))
    sc.pp.highly_variable_genes(temp, n_top_genes=2000, flavor='cell_ranger',duplicates='drop')
    tempGenes=temp.var['highly_variable']
    tempGenes.index = range(0,tempGenes.shape[0])#get reference id of gene,
    geneind = tempGenes[tempGenes == True].index

    #single cell rna
    scRNA = scRNA.to_numpy()
    scRNA = scRNA[:, geneind].T
    #m1_similarity = np.corrcoef(scRNA)
    m1_similarity = cosine_similarity(scRNA)
    tempIndex = np.where(m1_similarity < 0.5)
    m1_similarity[tempIndex] = 0
    adj = sp.csr_matrix(m1_similarity)
    adj = adj.astype('float32')
    print('adj size:', adj.shape)
    L = [laplacian(adj, normalized=True)]

    #string dataset
    '''
    adjpath = '/home/chenzhuo/PycharmProjects/GCN/gcndata/adjStringZhengsorted_21950.npz'
    adj = sp.load_npz(adjpath)
    adj = adj[geneind, :][:, geneind]
    adj = adj + sp.eye(adj.shape[0])
    adj = adj / np.max(adj)
    adj = adj.astype('float32')
    print('adj size:', adj.shape)
    L = [laplacian(adj, normalized=True)]
    '''

    '''
    #string dataset
    adj = sp.csr_matrix(scRNA.to_numpy())
    adj = adj[geneind, :][:, geneind]
    adj = adj + sp.eye(adj.shape[0])
    adj = adj / np.max(adj)
    adj = adj.astype('float32')
    print('adj size:', adj.shape)
    L = [laplacian(adj, normalized=True)]
    '''
    return geneind,L

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def test_cross(c_t, label, train_x, train_y, test_x, test_y, scRNA, seed=0):
    geneind, L = adjMatrix(scRNA)
    reproducibility(seed)
    data = torch.from_numpy(test_x).float().to(device)
    data_ = data[:, geneind]

    best_model, model = train_model(train_x, train_y, test_x, test_y, geneind, L, seed=seed)
    model.eval()
    _, pred, _ = model(data_, data, L)
    pred = pred.cpu().detach().numpy()
    a, b, c, d = score(pred, test_y)
    print('===out train===')
    print('mae:', np.round(a,3), 'rmse:', np.round(b,3), 'ccc:', np.round(c,3), 'ccc_mean:', np.round(d,3))

    # model = torch.load('model/model_'+str(seed) + '.pt')
    # _, pred, _ = model(data_, data, L)
    best_model.eval()
    _, pred, _ = best_model(data_, data, L)
    pred = pred.cpu().detach().numpy()
    a, b, c, d = score(pred, test_y)

    pred = pd.DataFrame(pred, columns=label)
    #pred.to_csv('Cross_val/'+str(seed)+'_'+str(c_t) +'.txt',sep='\t')
    print('===max===')
    print('mae:', np.round(a,3), 'rmse:', np.round(b,3), 'ccc:', np.round(c,3), 'ccc_mean:', np.round(d,3))

    #showheatmap(pred,test_y)

    return a, b

def test(train_x, train_y, test_x, test_y, scRNA, seed=0):
    geneind, L = adjMatrix(scRNA)
    reproducibility(seed)
    data = torch.from_numpy(test_x).float().to(device)
    data_ = data[:, geneind]

    best_model, model = train_model(train_x, train_y, test_x, test_y, geneind, L, seed=seed)
    model.eval()
    _, pred, _ = model(data_, data, L)
    pred = pred.cpu().detach().numpy()
    a, b, c, d = score(pred, test_y)
    print('===out train===')
    print('mae:', np.round(a,3), 'rmse:', np.round(b,3), 'ccc:', np.round(c,3), 'ccc_mean:', np.round(d,3))
    # pred = pd.DataFrame(pred)
    # pred.to_csv('/home/chenzhuo/PycharmProjects/chen_4_12/3/sars-cov-2/'+str(seed)+'.txt',sep='\t')

    # model = torch.load('model/model_'+str(seed) + '.pt')
    # _, pred, _ = model(data_, data, L)
    best_model.eval()
    _, pred, _ = best_model(data_, data, L)
    pred = pred.cpu().detach().numpy()
    a, b, c, d = score(pred, test_y)

    #draw res
    ##drop test_y=0
    # filtered_indices = np.nonzero(test_y)
    # test_y = test_y[filtered_indices]
    # pred = pred[filtered_indices]
    # filtered_indices = np.nonzero(pred)
    # test_y = test_y[filtered_indices]
    # pred = pred[filtered_indices]

    '''
    plt.scatter(test_y.reshape(-1), pred.reshape(-1), color='#0072B2')
    coefficients = np.polyfit(test_y.reshape(-1), pred.reshape(-1), 1)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(np.min(test_y), np.max(test_y), 100)
    y_fit = poly(x_fit)
    plt.plot(x_fit, y_fit, color='#E69F00', label='Fit Line')
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    # 去除上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 添加指标文本
    text_x = 0.1
    text_y = np.max(pred.reshape(-1))  # 指标文本位于拟合线上方2个单位
    #text = 'mae:'+ np.round(a, 3)+ 'rmse:'+ np.round(b, 3)+ 'ccc:'+ np.round(c, 3)
    text = 'ccc:'+ str(np.round(c, 3))
    plt.text(text_x, text_y, text, fontsize=12, color='black', ha='center')

    plt.show()
    '''

    pred = pd.DataFrame(pred)
    #pred.to_csv('/home/chenzhuo/PycharmProjects/chen_4_12/3/crc/gse81861_' + str(seed) + '.txt', sep='\t')
    pred.to_csv('/home/chenzhuo/PycharmProjects/chen_4_12/3/plat/sdy67_'+str(seed)+'.txt',sep='\t')
    #pred.to_csv('/home/chenzhuo/PycharmProjects/chen_4_12/3/sars-cov-2/' + str(seed) + '.txt', sep='\t')
    #test_y = pd.DataFrame(test_y)
    #test_y.to_csv('/home/chenzhuo/PycharmProjects/chen_4_12/3/cross_results/label_'+str(seed)+'_'+'mine',sep='\t')

    print('===max===')
    print('mae:', np.round(a,3), 'rmse:', np.round(b,3), 'ccc:', np.round(c,3), 'ccc_mean:', np.round(d,3))

    #showheatmap(pred,test_y)

    return a, b


def testGEGIPC(train_x, train_y, test_x, test_y, scRNA, output, seed=0):
    geneind, L = adjMatrix(scRNA)
    reproducibility(seed)
    data = torch.from_numpy(test_x).float().to(device)
    data_ = data[:, geneind]

    best_model, model = train_model(train_x, train_y, test_x, test_y, geneind, L, seed=seed)
    model.eval()
    _, pred, _ = model(data_, data, L)
    pred = pred.cpu().detach().numpy()
    a, b, c, d = score(pred, test_y)


    # model = torch.load('model/model_'+str(seed) + '.pt')
    best_model.eval()
    _, pred, _ = best_model(data_, data, L)
    pred = pred.cpu().detach().numpy()
    a, b, c, d = score(pred, test_y)


    pred = pd.DataFrame(pred)

    pred.to_csv(output+'pred.txt',sep='\t')


    #showheatmap(pred,test_y)

    return a, b
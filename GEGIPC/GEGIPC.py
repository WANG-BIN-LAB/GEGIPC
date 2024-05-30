import argparse
import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import MinMaxScaler
from test import testGEGIPC

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GEGIPC")
    parser.add_argument('--train_sample', '-trx', default='/', type=str, help='the path of train bulk dataset.')
    parser.add_argument('--test_sample', '-tex', default='/', type=str, help='the path of test bulk dataset.')
    parser.add_argument('--train_label', '-try', default='/', type=str, help='the path of train label dataset.')
    parser.add_argument('--test_label', '-tey', default='/', type=str, help='the path of test label dataset.')
    parser.add_argument('--scRNA', '-sc', default='/', type=str, help='the path of scRNA dataset.')
    parser.add_argument('--output', '-o', default='/', type=str, help='the path of outputed dataset.')

    args = parser.parse_args()

    train_x = pd.read(args.train_sample)
    train_y = pd.read(args.train_label)
    test_x = pd.read(args.test_sample)
    test_y = pd.read(args.test_label)


    scRNA = pd.read_table(args.scRNA, index_col=0)  # col=cells  raw=genes
    inter_gene = train_x.columns.intersection(scRNA.columns)
    scRNA = scRNA[inter_gene]
    label = test_x.var(axis=0) > 0.1
    scRNA = scRNA.iloc[:, label]
    test_x_new = np.zeros((test_x.shape[0], np.sum(label)))
    train_x_new = np.zeros((train_x.shape[0], np.sum(label)))
    k = 0
    for i in range(len(label)):
        if label[i] == True:
            test_x_new[:, k] = test_x[:, i]
            train_x_new[:, k] = train_x[:, i]
            k += 1
    train_x = train_x_new
    test_x = test_x_new
    train_x = np.log2(train_x + 1)
    test_x = np.log2(test_x + 1)
    # train_x, test_x = transformation(train_x, test_x)
    mms = MinMaxScaler()
    test_x = mms.fit_transform(test_x.T)
    test_x = test_x.T
    train_x = mms.fit_transform(train_x.T)
    train_x = train_x.T
    print(torch.__version__)
    print(device)
    seq = np.zeros((50, 2))
    for i in range(5):
        print(i)
        a, b = testGEGIPC(train_x, train_y, test_x, test_y, scRNA, output, seed=i)
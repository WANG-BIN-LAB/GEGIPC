import anndata
import torch
import pandas as pd
import numpy as np
import random
import scanpy as sc
import argparse
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler
from anndata import read_h5ad

from test import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def preprocess(trainingdatapath, testx=None, testy=None, testlabel='seq'):
    if (testx is not None) and (testy is not None):
        pbmc = read_h5ad(trainingdatapath)
        donorA = pbmc[pbmc.obs['ds'] == 'donorA']
        donorC = pbmc[pbmc.obs['ds'] == 'donorC']
        data6k = pbmc[pbmc.obs['ds'] == 'data6k']
        data8k = pbmc[pbmc.obs['ds'] == 'data8k']
        pbmc1 = pbmc[pbmc.obs['ds'] == 'sdy67']

        train_data = anndata.concat([donorA, donorC, data6k, data8k])
        test_x = pd.read_csv(testx, sep='\t', index_col=0)
        test_x = test_x[test_x.index != 'DZQV_PBMC']
        test_x = test_x.T
        test_y = pd.read_csv(testy, index_col=0)
        intersection_genes = list(test_x.index.intersection(train_data.var.index))
        adjpath = '/home/chenzhuo/data/some_data/pbmc/scRNA_pbmc_sortedGene.txt'
        scRNA = pd.read_table(adjpath, index_col=0)  # col=  raw
        scRNA = scRNA[intersection_genes]
        test_x = test_x.loc[intersection_genes]
        simuvar = list(train_data.var.index)
        intersection_gene_position = []
        for gene in intersection_genes:
            intersection_gene_position.append(simuvar.index(gene))
        selected = np.zeros((len(intersection_genes), len(train_data.X)))
        for i in range(selected.shape[0]):
            selected[i] = train_data.X.T[intersection_gene_position[i]]
        train_x = selected.T
        intersection_cell = list(test_y.columns.intersection(train_data.obs.columns))
        train_y = train_data.obs[intersection_cell].values
        ### re
        for i, values in enumerate(train_y):
            r_sum = np.sum(values)
            if r_sum == 0:
                pass
            else:
                train_y[i] = train_y[i] / r_sum
        ###
        test_y = test_y[intersection_cell]
        test_x = test_x.T
        test_x = test_x.values
        test_y = test_y.values
        ### re
        for i, values in enumerate(test_y):
            r_sum = np.sum(values)
            if r_sum == 0:
                pass
            else:
                test_y[i] = test_y[i] / r_sum
        ###
        assert test_x.shape[1] == train_x.shape[1]
        assert test_y.shape[1] == train_y.shape[1]
        return train_x, train_y, test_x, test_y, scRNA

    else:
        pbmc = read_h5ad(trainingdatapath)

        pbmc1 = pbmc[pbmc.obs['ds'] == 'sdy67']
        microarray = pbmc[pbmc.obs['ds'] == 'GSE65133']

        donorA = pbmc[pbmc.obs['ds'] == 'donorA']
        donorC = pbmc[pbmc.obs['ds'] == 'donorC']
        data6k = pbmc[pbmc.obs['ds'] == 'data6k']
        data8k = pbmc[pbmc.obs['ds'] == 'data8k']

        adjpath = '/home/chenzhuo/data/some_data/pbmc/scRNA_pbmc_sortedGene.txt'
        scRNA = pd.read_table(adjpath, index_col=0)  # col=  raw
        scRNA = scRNA[pbmc.var_names]

        if testlabel == 'seq':
            test = pbmc1
            train = anndata.concat([donorA, donorC, data6k, data8k])
            # train = anndata.concat([donorA, data8k])
        elif testlabel == 'microarray':
            test = microarray
            train = anndata.concat([donorA, donorC, data6k, data8k, pbmc1])
            # train = anndata.concat([donorA,data8k,pbmc1])
        elif testlabel == 'data6k':
            train = anndata.concat([donorA, donorC, data8k])
            #train = data6k
            temp_index = np.random.choice(range(8000),500,replace=False)
            data6k.obs['sub_test']=0
            for i in temp_index:
                data6k.obs.iloc[i,-1]=1
            test = data6k[data6k.obs['sub_test'] == 1]
            test.obs = test.obs.drop('sub_test',axis=1)
        elif testlabel == 'data8k':
            train = anndata.concat([donorA, donorC, data6k])
            #train = data6k
            temp_index = np.random.choice(range(8000),500,replace=False)
            data8k.obs['sub_test']=0
            for i in temp_index:
                data8k.obs.iloc[i,-1]=1
            test = data8k[data8k.obs['sub_test'] == 1]
            test.obs = test.obs.drop('sub_test',axis=1)
        elif testlabel == 'donorA':
            train = anndata.concat([data6k, data8k, donorC])
            #train = data6k
            temp_index = np.random.choice(range(8000),500,replace=False)
            donorA.obs['sub_test']=0
            for i in temp_index:
                donorA.obs.iloc[i,-1]=1
            test = donorA[donorA.obs['sub_test'] == 1]
            test.obs = test.obs.drop('sub_test',axis=1)
        elif testlabel == 'donorC':
            train = anndata.concat([data6k, data8k, donorA])
            # train = data6k
            temp_index = np.random.choice(range(8000), 500, replace=False)
            donorC.obs['sub_test'] = 0
            for i in temp_index:
                donorC.obs.iloc[i, -1] = 1
            test = donorC[donorC.obs['sub_test'] == 1]
            test.obs = test.obs.drop('sub_test', axis=1)

        train_y = train.obs.iloc[:, :-2].values
        test_y = test.obs.iloc[:, :-2].values

        label = test.X.var(axis=0) > -1
        ## variance cut off
        if testlabel == 'seq':
            label = test.X.var(axis=0) > 0.1
        elif testlabel == 'microarray':
            label = test.X.var(axis=0) > 0.01

        scRNA_new = scRNA.iloc[:, label]
        test_x_new = np.zeros((test.X.shape[0], np.sum(label)))
        train_x_new = np.zeros((train.X.shape[0], np.sum(label)))
        k = 0
        for i in range(len(label)):
            if label[i] == True:
                test_x_new[:, k] = test.X[:, i]
                train_x_new[:, k] = train.X[:, i]
                k += 1

        ####
        return train_x_new, train_y, test_x_new, test_y, scRNA_new

def preprocess_rosmap(trainingdatapath, testx, testy):
    trainset = read_h5ad(trainingdatapath)
    testset = pd.read_csv(testx, index_col=0,sep='\t')
    testlabel = pd.read_csv(testy, index_col=0)
    test_y = testlabel
    trainset.var_names = [s.upper() for s in trainset.var_names]

    intersection_genes = list(testset.columns.intersection(trainset.var_names))
    print(len(intersection_genes))

    scRNA = pd.read_table('/home/chenzhuo/data/some_data/brain/mousebrain_ref.txt',index_col=0)
    scRNA.columns = [s.upper() for s in scRNA.columns]

    train_x = pd.DataFrame(trainset.X, columns=trainset.var_names)
    train_x = train_x[intersection_genes].to_numpy()
    test_x = testset.loc[:,intersection_genes]
    test_x = test_x.to_numpy()

    #人类参考
    trainset.obs['Neurons'] = trainset.obs['ExNeurons']+trainset.obs['InNeurons']
    # find intersect cell proportions
    intersection_cell = list(test_y.columns.intersection(trainset.obs.columns))
    train_y = trainset.obs[intersection_cell]
    test_y = test_y[intersection_cell]
    ### refraction
    train_y = train_y.div(train_y.sum(axis=1), axis=0)
    train_y = train_y.fillna(0)

    ### variance cutoff
    label = test_x.var(axis=0) > 0.1
    #scRNA = scRNA.loc[intersection_cell, :]
    intersection_cell.extend(['ExNeurons', 'InNeurons'])
    intersection_cell.remove('Neurons')
    scRNA = scRNA.loc[intersection_cell, :]
    scRNA = scRNA[intersection_genes]
    scRNA_new = scRNA.iloc[:, label]
    test_x_new = np.zeros((test_x.shape[0], np.sum(label)))
    train_x_new = np.zeros((train_x.shape[0], np.sum(label)))
    k = 0
    for i in range(len(label)):
        if label[i] == True:
            test_x_new[:, k] = test_x[:, i]
            train_x_new[:, k] = train_x[:, i]
            k += 1

    return train_x_new, train_y.to_numpy(), test_x_new, test_y.to_numpy(), scRNA_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GEGIPC")
    parser.add_argument('--example', '-e', default='zhanged', type=str, help='test.')

    args = parser.parse_args()


    ###Cross validation
    if args.example=='zhanged':
        data=sc.read_h5ad('/home/chenzhuo/PycharmProjects/GCN/gcndata/sorted_Pbmc.h5ad')
        data_x=data.X
        data_y=data.obs.to_numpy()
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.25, random_state=0)
        # adjpath='/home/chenzhuo/PycharmProjects/GCN/gcndata/Filtered_DownSampled_SortedPBMC_data.csv'
        # scRNA = pd.read_csv(adjpath, index_col=0)  # col=cells  raw=genes
        # scRNA = scRNA[data.var_names]

        adjpath = '/home/chenzhuo/data/some_data/pbmc/scRNA_pbmc_sortedGene.txt'
        scRNA = pd.read_table(adjpath, index_col=0)  # col=cells  raw=genes
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
            a, b = test(train_x, train_y, test_x, test_y, scRNA, seed=i)

    elif args.example == 'newman':
        train_x, train_y, test_x, test_y, scRNA = preprocess('/home/chenzhuo/data/TAPE/2_Bulk/pbmc_data.h5ad',testlabel='microarray')
    elif args.example == 'monoca':
        train_x, train_y, test_x, test_y, scRNA = preprocess('/home/chenzhuo/PycharmProjects/deep_1/data/pbmc_data.h5ad',
                                                      '/home/chenzhuo/PycharmProjects/deep_1/data/pbmc2_samples.txt',
                                                      '/home/chenzhuo/PycharmProjects/deep_1/data/PBMC2.csv')

    elif args.example == 'sdy67' or args.example == 'gse107990':
        ###sdy67...
        for i in range(5):
            reproducibility(i)
            data6k = sc.read_h5ad('/home/chenzhuo/data/some_data/pbmc/500cells/data6k_500cells_8000samples.h5ad')
            data8k = sc.read_h5ad('/home/chenzhuo/data/some_data/pbmc/500cells/data8k_500cells_8000samples.h5ad')
            donorA = sc.read_h5ad('/home/chenzhuo/data/some_data/pbmc/500cells/donorA_500cells_8000samples.h5ad')
            donorA.obs['Unknown'] = np.zeros(donorA.obs.shape[0])
            donorC = sc.read_h5ad('/home/chenzhuo/data/some_data/pbmc/500cells/donorC_500cells_8000samples.h5ad')

            train = anndata.concat([donorA, donorC, data6k, data8k])
            train_x = pd.DataFrame(train.X,columns=train.var_names)
            train_y = train.obs

            if args.example == 'sdy67':
                test_x = pd.read_table('/home/chenzhuo/data/some_data/plat/sdy67_exp.txt', sep='\t', index_col=0).T#sdy67
                test_y = pd.read_table('/home/chenzhuo/data/some_data/plat/sdy67_labels.txt', sep='\t', index_col=0).T
                test_y.columns = ['CD4Tcells','CD8Tcells','NK','Monocytes','Bcells','Unknown']#sdy67

            else:
                test_x = pd.read_table('/home/chenzhuo/data/some_data/plat/gse107990_exp.txt', sep='\t', index_col=0).T
                test_y = pd.read_table('/home/chenzhuo/data/some_data/plat/gse107990_labels.txt', sep='\t', index_col=0)
                test_y.columns = ['CD4Tcells','CD8Tcells','Monocytes','NK','Bcells','Unknown']

            inter_cell = train_y.columns.intersection(test_y.columns)
            train_y = train_y[inter_cell].div(train_y.sum(axis=1),axis=0)
            test_y = test_y[inter_cell].div(test_y[inter_cell].sum(axis=1),axis=0)

            test_index = np.random.choice(range(test_y.shape[0]),50,replace= False)
            train_index = set(range(test_y.shape[0]))-set(test_index)
            test_index = list(test_index)
            train_index = list(train_index)

            train_y_ = test_y.iloc[train_index]
            test_y = test_y.iloc[test_index]
            train_y = pd.concat((train_y,train_y_))

            train_y = train_y.fillna(0).to_numpy()
            test_y = test_y.fillna(0).to_numpy()

            # train_y = train_y[inter_cell].to_numpy()
            # test_y = test_y[inter_cell].to_numpy()

            adjpath = '/home/chenzhuo/data/some_data/pbmc/scRNA_pbmc_sortedGene.txt'
            scRNA = pd.read_table(adjpath, index_col=0)  # col=cells  raw=genes

            inter_gene = train_x.columns.intersection(test_x.columns).intersection(scRNA.columns)
            train_x = train_x[inter_gene]
            test_x = test_x[inter_gene]
            test_x = test_x.groupby(test_x.columns, axis=1).mean()# equal gene name get mean value
            train_x = train_x[inter_gene]
            test_x = test_x[inter_gene]

            train_x_ = test_x.iloc[train_index]
            test_x = test_x.iloc[test_index]
            train_x = pd.concat((train_x,train_x_))
            train_x = train_x.to_numpy()
            test_x = test_x.to_numpy()

            scRNA = scRNA[inter_gene]
            # label = test_x.var(axis=0) > 0.1
            # scRNA = scRNA.iloc[:, label]
            # test_x_new = np.zeros((test_x.shape[0], np.sum(label)))
            # train_x_new = np.zeros((train_x.shape[0], np.sum(label)))
            # k = 0
            # for i in range(len(label)):
            #     if label[i] == True:
            #         test_x_new[:, k] = test_x[:, i]
            #         train_x_new[:, k] = train_x[:, i]
            #         k += 1
            # train_x = train_x_new
            # test_x = test_x_new
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
            seq = np.zeros((50,2))

            print(i)
            a,b = test(train_x, train_y, test_x, test_y, scRNA, seed=i)



        ###CRC
        else:
            ##81861
            train = sc.read_h5ad('/home/chenzhuo/data/some_data/crc/gse81861_train.h5ad')
            train_x = pd.DataFrame(train.X,columns=train.var_names)
            train_y = train.obs
            test_x = pd.read_table('/home/chenzhuo/data/some_data/crc/gse81961_samples.txt',index_col=0)
            test_y = pd.read_table('/home/chenzhuo/data/some_data/crc/gse81961_labels.txt',index_col=0)
            adjpath = '/home/chenzhuo/data/some_data/crc/gse81861_scRNA.txt'



            inter_cell = train_y.columns.intersection(test_y.columns)
            train_y = train_y[inter_cell].div(train_y.sum(axis=1),axis=0)
            test_y = test_y[inter_cell]
            test_y = test_y.div(test_y.sum(axis=1),axis=0)
            train_y = train_y.fillna(0).to_numpy()
            test_y = test_y.fillna(0).to_numpy()

            scRNA = pd.read_table(adjpath, index_col=0)  # col=cells  raw=genes

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
            seq = np.zeros((50,2))
            for i in range(5):
                print(i)
                a,b = test(train_x, train_y, test_x, test_y, scRNA, seed=i)

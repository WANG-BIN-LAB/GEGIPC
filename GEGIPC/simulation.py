import pandas as pd
import numpy as np
import anndata
from tqdm import tqdm
from numpy.random import choice
import re
import scanpy as sc
import argparse

def simulate(scRNA_path, out_path, cell_nums = 500, sample_nums = 3000, d_prior=''):
    if type(scRNA_path) is str:#cell*gene
        if scRNA_path.partition('.')[-1]=='csv':
            sc_data = pd.read_csv(scRNA_path,index_col=0,sep='\t')
        if scRNA_path.partition('.')[-1] == 'txt':
            sc_data = pd.read_table(scRNA_path, index_col=0, sep='\t')
            # sc_data.rename(columns=lambda x: re.sub('\.\d+', '', x), inplace=True)
            # sc_data = sc_data.T
            # sc_data = sc_data.iloc[:4722, ]
        if scRNA_path.partition('.')[-1] == 'h5':
            data = sc.read_10x_h5(scRNA_path)
            temp = data.X
            import pandas as pd
            sc_data = pd.DataFrame(temp.toarray(), index=data.obs.index, columns=data.var_names)
            sc_data['patient'] = [x[0] for x in sc_data.index.str.split('_')]
            index = ['C113', 'C125', 'C147', 'C157', 'C122', 'C163', 'C134', 'C152', 'C168', 'C155']
            sc_data = sc_data[sc_data['patient'].isin(index)]
            sc_data = sc_data.drop('patient',axis=1)
            label = pd.read_table('/home/chenzhuo/data/some_data/crc/crc10x_tSNE_cl_global.tsv')
            dic = dict(zip(label['NAME'], label['ClusterTop']))
            del temp
            temp = [x for x in sc_data.index]
            index_new = [dic.get(item, item) for item in temp]
            sc_data.index = index_new
    elif type(scRNA_path) is pd.DataFrame:
        pass
    else:
        raise Exception("Please check the format of single-cell data!")
    print('Reading dataset is done')
    sc_data.fillna(0)
    # sc_data = sc_data.loc[['alpha','beta','delta','gamma']]
    #sc_data=sc_data.rename({'Dendritic':'Unknown','Neutrophil':'Unknown'})
    sc_data['celltype'] = sc_data.index
    sc_data.index = range(len(sc_data))

    num_celltype = len(sc_data['celltype'].value_counts())
    genename = sc_data.columns[:-1]
    celltype_groups = sc_data.groupby('celltype').groups
    sc_data.drop(columns='celltype', inplace=True)

    sc_data = sc_data.values
    sc_data = np.ascontiguousarray(sc_data, dtype=np.float32)#将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。

    if d_prior == '':
        print('Generating cell fractions using Dirichlet distribution without prior info (actually random)')
        prop = np.random.dirichlet(np.ones(num_celltype), sample_nums)
    elif d_prior is not None:
        print('Using prior info to generate cell fractions in Dirichlet distribution')
        assert len(d_prior) == num_celltype, 'dirichlet prior is a vector, its length should equals ' \
                                             'to the number of cell types'
        prop = np.random.dirichlet(d_prior, sample_nums)
        print('Dirichlet cell fractions is generated')

    # make the dictionary
    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)
    # precise number for each celltype
    cell_num = np.floor(cell_nums * prop)

    # start sampling
    sample = np.zeros((prop.shape[0], sc_data.shape[1]))
    allcellname = celltype_groups.keys()
    print('Sampling cells to compose pseudo-bulk data')
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data[select_index].sum(axis=0)

    prop = pd.DataFrame(prop, columns=celltype_groups.keys())
    simulated_dataset = anndata.AnnData(X=sample,
                                        obs=prop,
                                        var=pd.DataFrame(index=genename))
    print('Sampling is done')
    if out_path is not None:
        simulated_dataset.write_h5ad(out_path + 'gse178341_train.h5ad')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulated gene expression data")
    parser.add_argument('--scRNA_path', '-sc', default='/home/chenzhuo/data/some_data/crc/GSE178341_crc10x_full_c295v4_submit.h5', type=str, help='the path of scRNA gene expression.')
    parser.add_argument("--out_path", '-o', default='/home/chenzhuo/data/some_data/crc/', type=str, help="the path of output-path.")
    parser.add_argument("--cell_nums", '-c', default=500, type=int, help="Number of cells.")
    parser.add_argument("--sample_nums", '-s', default=3000, type=int, help="Number of samples.")
    args = parser.parse_args()

    simulate(args.scRNA_path,args.out_path,args.cell_nums,args.sample_nums)
# GEGIPC

## Overview 

Cell type deconvolution is a computational method employed to discern the relative abundance of distinct cell types within tissue samples. Single-cell sequencing is typically favored for obtaining the proportions of cell types. Nevertheless, the current costliness of single-cell sequencing prohibits its application in clinical studies encompassing a sizable cohort. This study introduces a multimodal deep learning model, integrating deep neural networks and graph convolutional networks, for predicting cell composition by organizing gene expression and gene interrelationships (GEGIPC).This Python script is used to train, validate, and test deep learning models for cell deconvolution. We use PyTorch to build GEGIPC. You can set hyperparameters as needed. Please refer to the following description. The reference single-cell expression data or gene expression data can be saved in a.txt file format. This feature should be in a tab-separated format for scripts to parse data.

## Requirement

```
argparse
numpy
pandas
random
scanpy >= 1.9.3
scipy >= 1.1.0
scikit-learn >= 1.0.2
torch == 1.8.0+cu111
```
## Usage 
### Sample simulation
`python simulation.py -sc scRNA_path -o out_path -c cell_nums -s sample_nums`
### Run the examples
`python example -e example`
### Using GEGIPC
`python GEGIPC.py -trx train_sampke -tex test_sample -try train_label -sc scRNA -o output`
The parameter description is as follows：
```
Params:
  --train_sample  -trx  The path of train bulk dataset
  --test_sample  -tex  The path of test bulk dataset.
  --train_label  -try  The path of train label dataset.
  --scRNA  -sc  The path of scRNA dataset.
  --output  -o  The path of outputed dataset.
```

## Data Specification

All training, validation, test should follow specification to be parsed correctly by GEGIPC:

* The model consists of four types of data, including sample bulk gene expression data, sample bulk cell composition data, predicted sample bulk gene expression data, and used to construct interacting single-cell gene data.

* They should be `.txt` format.

* For feature column, each dimension of features in columns should be delimited with tab (`\t`)


## Reproduction instructions

The above scripts can reproduce the quantitative results in our manuscript based on our provided data.

## Contact 

liying01@tyut.edu.cn

chenzhuo0648@link.tyut.edu.cn

wangbin01@tyut.edu.cn

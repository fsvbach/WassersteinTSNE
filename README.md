# WassersteinTSNE

see paper at www.arXive.org/WassersteinTSNE

## Installation

You can install this package via 

`pip install WassersteinTSNE`

## Basic Usage

First import the package with 

`import WassersteinTSNE as WT`

### Data 

The data should be provided in either of two ways:

1. As a pandas dataframe where the index indicates which sample belongs to which units
2. As a numpy ndarray where each line corresponds to a sample **and** a list of unit ids

If you don't have a dataset you can generate a toy dataset by running

`dataset = WT.ToyDataset()`

or create a random HGMM

```
HGMM = WT.HierarchicalGaussianMixture()
dataset = HGMM.generate_data()
```

By default that creates a HGMM with K=5 classes. This corresponds to a pandas dataframe with N=250 units and M=15 samples each. Each sample has F=2 features. 

### Visualization

To visualize the generated (two-dimnesional) HGMM you can run

`WT.plotMixture(HGMM)`

### Gaussian Wasserstein t-SNE

The straight forward way to embed your hierarchical data is 

`embedding = WT.TSNE(seed=17, w=.5)`

or do the procedure step by step with

```
Gaussians = WT.dataset2Gaussians()
WSDM      = WT.WSDM(Gaussians)
embedding = WT.GaussianTSNE(WSDM, w=0.5)
```


## Exact Wasserstein Distances

Despite its complexity it is possibly to compute exact Wasserstein distances of a dataset with

`X = WT.WassersteinDistanceMatrix(dataset)`

This




To visualize the evolution of the embedding with increasing `w` you can use the implemented method


## Evaluation

You can use the evaluation ethods of the Leiden algorithm and kNN accuracy with



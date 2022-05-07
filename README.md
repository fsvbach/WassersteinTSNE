# WassersteinTSNE

see paper at www.arXiv.org/WassersteinTSNE

## Installation

You can install this package via 

`pip install WassersteinTSNE`.

## Basic Usage

First import the package with 

`import WassersteinTSNE as WT`

### Data 

The data should be provided in either of two ways:

1. As a `pd.DataFrame` where the index indicates which sample belongs to which units
2. As a `np.ndarray` where each line corresponds to a sample **and** a list of unit ids

If you don't have a dataset at hand you can generate a toy dataset by running

`dataset = WT.ToyDataset()`

or create a random HGMM

```
HGMM = WT.HierarchicalGaussianMixture(seed=67)
dataset = HGMM.generate_data()
```

By default that creates a HGMM with K=4 classes. This corresponds to a `pd.DataFrame` with N=100 units and M=30 samples each. If each sample has F=2 features, you can visualize the generated HGMM by

`WT.plotMixture(HGMM)`

### Gaussian Wasserstein t-SNE

The straight forward way to embed your hierarchical dataset is 

`embedding = WT.TSNE(seed=67, w=.5)`

or do the procedure step by step with

```
Gaussians = WT.Dataset2Gaussians(dataset)
GWD       = WT.GaussianWassersteinDistance(Gaussians)
embedding = WT.ComputeTSNE(GWD.matrix(w=0.5), seed=67)
```

Note that the second way offers many tools for further analysis, e.g. you can obtain the distance matrix for any value of `w` with 

`D = GWD.matrix(w=0.5)`.

By adjusting `w` you can put emphasis on the means or covariance matrices of the units: 

`embedding = WTSNE.fit(w=0.7)` 

All embeddings are returned as a `pd.DataFrame`, which can be visualized with

`WT.embedScatter(embedding, title='DemoEmbedding')`


## Exact Wasserstein Distances

Despite its complexity it is possibly to compute exact Wasserstein distances of a dataset with

`X = WT.WassersteinDistanceMatrix(dataset)`

This yields the NxN distance matrix which can then be embedded with

`embedding = WT.ComputeTSNE(D)`

A shortcut for this procedure is given by

`embedding = WT.TSNE(seed=67, method='exact')`


## Evaluation

You can use the evaluation ethods of the Leiden algorithm and kNN accuracy with

### knn accuracy


### Leiden clustering

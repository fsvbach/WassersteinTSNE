# WassersteinTSNE

This package provides the methods described in the _Wasserstein t-SNE_ paper at [www.arXiv.org/WassersteinTSNE](). It is a doesn't contain the research code, but

To reproduce the figures in the paper, please also check the repository [wassersteinTSNE-paper](www.github.com/fsvbach/wassersteinTSNE-paper). 



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
HGM = WT.HierarchicalGaussianMixture(seed=67)
dataset = HGM.generate_data()
```

By default that creates a HGMM with K=4 classes. This corresponds to a `pd.DataFrame` with N=100 units and M=30 samples each. If each sample has F=2 features, you can visualize the generated HGMM by

`WT.plotMixture(HGM)`

### Gaussian Wasserstein t-SNE

The straight forward way to embed your hierarchical dataset is 

`embedding = WT.TSNE(dataset, seed=67, w=0.5)`

or do the procedure step by step with

```
Gaussians = WT.Dataset2Gaussians(dataset)
GWD       = WT.GaussianWassersteinDistance(Gaussians)
embedding = WT.ComputeTSNE(GWD.matrix(w=0.5), seed=67)
```

Note that the second way offers many tools for further analysis, e.g. you can obtain the distance matrix for any value of `w` with 

`D = GWD.matrix(w=0.5)`.

By adjusting `w` you can put emphasis on the means or covariance matrices of the units: 

`embedding = WT.TSNE(seed=67, w=0.7)` 

All embeddings are returned as a `pd.DataFrame`, which can be visualized with

`WT.embedScatter(embedding, title='DemoEmbedding')`

If you have defined classes, you can pass a dictionary that maps the unit ids to their class

`WT.embedScatter(embedding, labeldict=HGM.labeldict())`

to color the units according to their class.

## Exact Wasserstein Distances

It is possibly to compute the exact Wasserstein distances of a dataset as well. Depending on the number of units this can take some time. However, for the dataset in `WT.ToyDataset()` the computation of the pairwise distance matrix should take less than 8min on a desktop computer by running

`D = WT.WassersteinDistanceMatrix(dataset)`

This yields the NxN distance matrix as a `pd.DataFrame` which can then be embedded with

`embedding = WT.ComputeTSNE(D)`

A shortcut for this procedure is provided with

`embedding = WT.TSNE(dataset, method='exact')`


## Evaluation

We implemented two methods to evaluate the embedding of a hierarchical dataset. 

### kNN Accuracy



### Leiden clustering

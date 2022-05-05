#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:35:42 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np

from openTSNE import TSNE as openTSNE
from sklearn.manifold import TSNE as skleTSNE
from sklearn.metrics import accuracy_score, cluster
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
import igraph as ig 
import leidenalg as la

from .Distributions import RotationMatrix

class GaussianTSNE:
    def __init__(self, GWD, seed=None, perplexity=30, sklearn=False):
        self.GWD     = GWD
        self.sklearn = sklearn
        self.seed    = seed
        self.perplexity = perplexity

    def fit(self, w, trafo=None):

        if self.sklearn:
            tsne = skleTSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=self.seed)
            embedding = tsne.fit_transform(self.GWD.matrix(w=w))
        else:
            tsne = openTSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=self.seed,
                        perplexity = self.perplexity)
            embedding = tsne.fit(self.GWD.matrix(w=w))
        
        if trafo is None:
            trafo = np.eye(2)

        embedding =  pd.DataFrame( embedding @ trafo, 
                     index=self.GWD.index,
                     columns = ['x','y'])
        return embedding
    
    def knn_accuracy(self, w, labeldict, k=5):
        embedding = self.fit(w)
        embedding.index = embedding.index.to_series().map(labeldict)
        kNN    = KNeighborsClassifier(k+1)
        kNN.fit(embedding.values, embedding.index)
        test = kNN.predict(embedding.values)
        return accuracy_score(test, embedding.index)*(k+1)/k-1/k

    def adjusted_rand_index(self, w, labeldict, k=5, res = 0.08):
        A = kneighbors_graph(self.GWD.matrix(w=w), k)
        sources, targets = A.nonzero()
        G = ig.Graph(directed=True)
        G.add_vertices(A.shape[0])
        edges = list(zip(sources, targets))
        G.add_edges(edges)
        clustering = la.find_partition(G, la.RBConfigurationVertexPartition, 
                                          resolution_parameter = res, seed=self.seed)
        groundtruth = self.fit(w).index.to_series().map(labeldict)
        return cluster.adjusted_rand_score(clustering.membership, groundtruth)
    
class NormalTSNE:
    def __init__(self, seed=None, sklearn=False):
        self.sklearn = sklearn
        self.seed    = seed
        
    def fit(self, dataset, trafo=None):
        if self.sklearn:
            tsne = skleTSNE(random_state=self.seed)
            embedding = tsne.fit_transform(dataset.values)
        else:
            tsne = openTSNE(random_state=self.seed, initialization='random')
            embedding = tsne.fit(dataset.values)
            
        if trafo is None:
            trafo = np.eye(2)
        embedding =  pd.DataFrame(embedding @ trafo, 
                     index=dataset.index,
                     columns = ['x','y'])
        return embedding
    
class WassersteinTSNE:
    def __init__(self, seed=None, perplexity=30, sklearn=False):
        self.sklearn = sklearn
        self.seed    = seed
        self.perplexity = perplexity
        
    def fit(self, dataset, trafo=None):
        if self.sklearn:
            tsne = skleTSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=self.seed)
            embedding = tsne.fit_transform(dataset.values)
        else:
            tsne = openTSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=self.seed,
                        perplexity = self.perplexity)
            embedding = tsne.fit(dataset.values)
        
        if trafo is None:
            trafo = np.eye(2)
        embedding =  pd.DataFrame(embedding @ trafo, 
                     index=dataset.index,
                     columns = ['x','y'])
        return embedding
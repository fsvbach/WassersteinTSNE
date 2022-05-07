#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:18:09 2022

@author: bachmann
"""

from sklearn.metrics import accuracy_score, cluster
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
import igraph as ig 
import leidenalg as la

def knn_accuracy(embedding, labeldict, k=5):
    embedding.index = embedding.index.to_series().map(labeldict)
    kNN    = KNeighborsClassifier(k+1)
    kNN.fit(embedding.values, embedding.index)
    test = kNN.predict(embedding.values)
    return accuracy_score(test, embedding.index)*(k+1)/k-1/k

def adjusted_rand_index(D, labeldict, k=5, res = 0.08, seed=None):
    kNNgraph = kneighbors_graph(D, k)
    sources, targets = kNNgraph.nonzero()
    G = ig.Graph(directed=True)
    G.add_vertices(kNNgraph.shape[0])
    G.add_edges(list(zip(sources, targets)))
    clustering = la.find_partition(G, la.RBConfigurationVertexPartition, 
                                      resolution_parameter = res, seed=seed)
    groundtruth = D.index.to_series().map(labeldict)
    return cluster.adjusted_rand_score(clustering.membership, groundtruth)

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

def knnAccuracy(embedding, labeldict, k=5):
    """
    Input:
    embedding: pd.DataFrame
    labeldict: dictionary that maps the index  of embedding to labels
    
    Returns
    -------
    Accuracy as float.
    
    Note: Since it is not wanted that the label of the test point is used for classification, we multiply the restulting accuracy by the factor of (k+1)/k and subtract 1/k.
    """
    labels = embedding.index.to_series().map(labeldict)
    kNN    = KNeighborsClassifier(k+1)
    kNN.fit(embedding.values, labels)
    estmte = kNN.predict(embedding.values)
    return accuracy_score(estmte, labels)*(k+1)/k-1/k

def LeidenClusters(D, labeldict, k=5, res = 0.08, seed=None):
    kNNgraph = kneighbors_graph(D, k)
    sources, targets = kNNgraph.nonzero()
    G = ig.Graph(directed=True)
    G.add_vertices(kNNgraph.shape[0])
    G.add_edges(list(zip(sources, targets)))
    clustering = la.find_partition(G, la.RBConfigurationVertexPartition, 
                                      resolution_parameter = res, seed=seed)
    labels = D.index.to_series().map(labeldict)
    return cluster.adjusted_rand_score(clustering.membership, labels)
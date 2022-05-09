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

from .Distances import GaussianWassersteinDistance, WassersteinDistanceMatrix
from .utils import Dataset2Gaussians

def ComputeTSNE(D, seed=None, perplexity=30, sklearn=False, trafo=None):
    if sklearn:
        tsne = skleTSNE(metric='precomputed', 
                    square_distances=True, 
                    random_state=seed)
        embedding = tsne.fit_transform(D)
    else:
        tsne = openTSNE(metric='precomputed', 
                    initialization ='random', 
                    negative_gradient_method='bh',
                    random_state  =seed,
                    perplexity    = perplexity)
        embedding = tsne.fit(D)
    
    if trafo is None:
        trafo = np.eye(2)

    embedding =  pd.DataFrame( embedding @ trafo, 
                 index=D.index,
                 columns = ['x','y'])
    return embedding

    
def TSNE(X, y=None, seed=None, method='gaussian', w=.5):
    if y:
        X = pd.DataFrame(X, index=y)
    assert isinstance(X, pd.DataFrame)
    
    if method == 'gaussian':
        Gaussians = Dataset2Gaussians(X)
        GWD       = GaussianWassersteinDistance(Gaussians)
        embedding = ComputeTSNE(GWD.matrix(w=w), seed=seed)
        
    elif method == 'exact':
        D = WassersteinDistanceMatrix(X)
        embedding = ComputeTSNE(D, seed=seed)
    
    else:
        raise AssertionError("Please type method='gaussian' or method='exact'")
    
    return embedding
        
    
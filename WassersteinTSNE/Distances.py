#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:57:05 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.optimize import linprog
from .utils import arr2cov, Timer

def EuclideanDistance(A,B):
    N1 = np.linalg.norm(A, ord=2, axis=1).reshape(-1,1)**2 
    N2 = np.linalg.norm(B, ord=2, axis=1).reshape(1,-1)**2 
    N3 = -2 * np.inner(A,B)
    D  = N1 + N2 + N3
    D[np.where(D<0)] = 0
    return np.sqrt(D)
    
def ConstraintMatrix(n,m):
    N = np.repeat(np.identity(n,dtype=int), m, axis=1)
    M = np.hstack([np.identity(m, dtype=int)]*n)
    return np.vstack([N,M])

def SparseConstraint(n,m):
    row  = np.repeat(np.arange(n), m)
    col  = np.arange(n*m) 
    data = np.ones(n*m, dtype=int)
    N = sp.csr_matrix((data, (row, col)), 
                   shape=(n, n*m))
    M = sp.hstack([sp.dia_matrix((np.ones(m, dtype=int), 0), shape=(m,m))]*n)
    return sp.vstack([N,M], format='csr')

def PairwiseWassersteinDistance(U, V, p=2):
    uniqueRowsU, occurCountU = np.unique(U, axis=0, return_counts=True)
    uniqueRowsV, occurCountV = np.unique(V, axis=0, return_counts=True)
    
    D = EuclideanDistance(uniqueRowsU, uniqueRowsV)**p
    n, m = len(uniqueRowsU), len(uniqueRowsV)
  
    A = SparseConstraint(n,m)
    b = np.concatenate([occurCountU/sum(occurCountU), occurCountV/sum(occurCountV)])
    c = D.reshape(-1)

    return linprog(-b, A.T, c, bounds=[None, None], method='highs')
    
def WassersteinDistanceMatrix(dataset, timer=True):
    logs = Timer('WSDM', output=False)  
    if isinstance(dataset.index , pd.MultiIndex):
        dataset.index = dataset.index.get_level_values(0)
        
    unit_ids = dataset.index.unique()
    N = len(unit_ids )
    K = np.zeros((N,N))
    
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            opt_res = PairwiseWassersteinDistance(dataset.loc[unit_ids[i]], 
                                                  dataset.loc[unit_ids[j]])
            K[i,j] = -opt_res.fun
            
            if k%250 == 0 and timer:
                logs.add(f'Completed {k} of {N*(N-1)/2}')
            k+=1
    
    K = np.sqrt(K + K.T)
    
    return pd.DataFrame(K, index=unit_ids , columns=unit_ids)


class GaussianWassersteinDistance:
    def __init__(self, Gaussians, fast_approx=False):
        sqrts = []
        means = []
        covs  = []
        
        self.index = Gaussians.index
        for G in Gaussians:            
            sqrts.append(G.cov.sqrt())
            covs.append(G.cov)
            means.append(G.mean)
            
        self.EDM = self.EuclideanDistanceMatrix(np.stack(means))
        
        if fast_approx:
            self.CDM = self.FrobeniusDistanceMatrix(np.stack(sqrts))
        else:
            self.CDM = self.CovarianceDistanceLoop(covs)
         
    def EuclideanDistanceMatrix(self, X):
        norms  = np.linalg.norm(X, axis=1, ord=2).reshape((len(X),1))**2
        matrix = norms + norms.T - 2 * X@X.T
        return matrix - matrix.min()
    
    def FrobeniusDistanceMatrix(self, X):
        norms = np.linalg.norm(X, ord='fro', axis=(1, 2)).reshape((len(X),1))**2
        matrix = norms + norms.T - 2 * np.tensordot(X,X, axes=([1,2],[1,2]))
        return matrix - matrix.min()
    
    def PairwiseCovarianceDistance(self, cov1, cov2):
        tmp = cov2.sqrt() @ cov1.array() @ cov2.sqrt()
        tmp = arr2cov(tmp)
        tmp = cov1.array() + cov2.array() - 2 * tmp.sqrt()
        return np.sum(np.diag(tmp))

    def CovarianceDistanceLoop(self, covs):
        N = len(covs)
        K = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                K[i,j] = self.PairwiseCovarianceDistance(covs[i], covs[j])
        return K + K.T
    
    def matrix(self, w=0.5):
        K =  np.sqrt(2-4*(w-0.5)**2)*np.sqrt((1-w)*self.EDM + w*self.CDM)
        return pd.DataFrame(K, index=self.index , columns=self.index)
    
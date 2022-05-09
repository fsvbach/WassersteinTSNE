#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:20:51 2021

@author: fsvbach
"""

from scipy.stats import special_ortho_group, wishart
from scipy.linalg import eigh
import numpy as np
import pandas as pd
import time

def RotationMatrix(degree):
    degree = np.deg2rad(degree)
    return np.array([[np.cos(degree),-np.sin(degree)],
                     [np.sin(degree), np.cos(degree)]]) 

def MirrorMatrix(vector):
    x,y = vector
    return np.array([[x**2-y**2,2*x*y],
                     [2*x*y, y**2-x**2]]) / np.linalg.norm(vector)**2

class CovarianceMatrix:
    def __init__(self, P=RotationMatrix(45), s=np.array([1,4])):
        '''
        stores eigen-decomposition of covariance matrix
        ----------
        P : np.ndarray
            orthogonal matrix.
        s : np.array
            eigenvalues in array.
        '''       
        self.P = P
        self.s = s
        
    def diagonalize(self):
        self.s  = np.diag(self.array())
        self.P  = np.eye(len(self.s))
        
    def normalize(self):
        A = self.array()
        diagonal = np.sqrt(np.diag(A)) + 0.00000001
        A = (A/diagonal).T / diagonal
        self.s, self.P = eigh(A)
        self.s[np.where(self.s<0)]=0
            
    def array(self):
        return self.P@np.diag(self.s)@self.P.T
    
    def sqrt(self):
        return self.P@np.diag(np.sqrt(self.s))@self.P.T
    
    def shape(self, std=1):
        assert len(self.s) == 2
        angle         = np.degrees(np.arctan2(*self.P[:,0][::-1]))
        width, height = np.sqrt(self.s)*std*2
        return width, height, angle 

def arr2cov(array):
    s, P = eigh(array)
    s[np.where(s<0)]=0
    assert np.all(s>=0)
    return CovarianceMatrix(P, s)  

class WishartDistribution:
    def __init__(self, nu=2, scale=CovarianceMatrix()):
        self.nu = nu
        self.scale  = scale
        
    def shape(self, std=1):
        return self.scale.shape(std=std)
        
class GaussianDistribution:
    def __init__(self, mean=np.array([1,0]), cov=CovarianceMatrix()):
        self.mean = mean
        self.cov  = cov

    def estimate(self, data):
        N = len(data)
        assert N > 1
        self.mean = np.mean(data, axis=0)
        self.cov  = arr2cov((data-self.mean).T @ (data-self.mean) / (N-1))
        return GaussianDistribution(self.mean, self.cov)
    
    def shape(self, std=1):
        width, height, angle = self.cov.shape(std=std)
        return self.mean, width, height, angle

    def samples(self, size=20, seed=None):
        return np.random.default_rng(seed=seed).multivariate_normal(mean = self.mean, 
                                                  cov  = self.cov.array(),
                                                  size = size) 

def Dataset2Gaussians(dataset, diagonal=False, normalize=False):
    Gaussians = []
    names    = []
    for name, data in dataset.groupby(level=0):
        G = GaussianDistribution()
        G.estimate(data.values)
        if diagonal:
            G.cov.diagonalize()
        elif normalize:
            G.cov.normalize()
        names.append(name)
        Gaussians.append(G)
    return pd.Series(Gaussians, index=names)

class RandomGenerator:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)
    
    def RandomSeed(self):
        return self.UniformInteger(upper=10000)[0]
    
    def UniformInteger(self, lower=0, upper=10, size=1):
        return self.generator.integers(lower, upper, size)
    
    def UniformVector(self, dim=2, upper=1, size=1):
        samples = self.generator.random((size, dim)) - 0.5 
        return 2*samples*upper
    
    def OrthogonalMatrix(self, dim=2, size=1):
        return special_ortho_group.rvs(dim=dim, random_state=self.RandomSeed(), size=size)
    
    def UniformCovariance(self, dim=2, maxstd=1):
        P = self.OrthogonalMatrix(dim=dim)
        s = (self.generator.random(size=dim) * maxstd)**2
        return CovarianceMatrix(P,s)
        
    def GaussianSamples(self, Gaussian, size=1):
        return self.generator.multivariate_normal(mean = Gaussian.mean, 
                                                  cov  = Gaussian.cov.array(),
                                                  size = size)
    def WishartSamples(self, Wishart, size=1):
        arrays = wishart.rvs(Wishart.nu, Wishart.scale.array(), random_state=self.RandomSeed(), size=size)
        return [arr2cov(arr) for arr in arrays]

class Timer:
    def __init__(self, name, dec=3, output=True):
        self.name      = name
        self.dec       = 3
        self.output    = output
        self.start     = self.time()
        self.last_time = self.start
        self.date      = self.date()
        self.log       = [f"Started '{name}' at {self.date}\n"]
  
    def date(self):
        t = time.localtime()
        return time.strftime("%m-%d-%H-%M-%S", t)
    
    def time(self):
        return round(time.perf_counter(), self.dec)
    
    def total_time(self):
        return round(self.time() - self.start, self.dec)
    
    def add(self, infomsg):
        time = self.time() - self.last_time
        msg  = f'{infomsg} in {round(time, self.dec)}s. (Total: {self.total_time()}s)\n'
        self.last_time = self.time()
        self.log.append(msg)
        print(msg)
    
    def result(self, result):
        self.log.append(result)
        print(result)

    def finish(self, location):
        infomsg = f"Succesfully finished '{self.name}' in {self.total_time()}s"
        print(infomsg)
        
        if self.output:
            for msg in reversed(self.log):
                infomsg = msg + '\n' + infomsg
            file = open(location, "a")
            file.write(infomsg)
            file.write(f"\n\n{'-'*75}\n")
            file.close() 
        
    

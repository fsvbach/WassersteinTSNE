#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:20:51 2021

@author: fsvbach
"""

naming = {0: 'Euclidean', 0.5: 'Wasserstein', 1: 'Covariance'} 

from .Distributions import CovarianceMatrix, arr2cov
from scipy.stats import special_ortho_group, wishart
import numpy as np
import time


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
        
    

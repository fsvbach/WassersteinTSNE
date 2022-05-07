#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:04:54 2021

@author: fsvbach
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow
from matplotlib.legend_handler import HandlerPatch

def embedScatter(embedding, title='', size=1, ax=None):
    if not ax:
        ax = plt.gca()
        
    for label, data in embedding.groupby(level=0):
        X, Y = data['x'], data['y']
        ax.scatter(X, Y, s=size, label=label)

    ax.set_xticks([], minor=[])
    ax.set_yticks([], minor=[])
    # for spine in ['bottom', 'top', 'left', 'right']:
    #     ax.spines[spine].set_linestyle("dashed")
    ax.set_title(title)

def HandlerArrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

class HandlerEllipseRotation(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       ydescent, xdescent, height, width, fontsize, trans):
        center = 0.5 * height, 0.5 * width
        p = Ellipse(xy=center, width=orig_handle.width,
                             height=orig_handle.height,
                             angle=orig_handle.angle, linewidth=.5)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
    
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
        
def plotMixture(mixture, std=2, ax=None):
    fig = None
    if not ax:
        fig, ax = plt.subplots(figsize=(7,7))
    
    # covlabel = rf'{std}-$\sigma$ class covariance'
    covlabel = 'Classes'
    dataset  = mixture.data.groupby(level=0).mean().sample(frac=1, random_state=43)
    dataset.index = dataset.index.to_series().map(mixture.labeldict())
    

    if std > 0:
        for Gaussian in mixture.ClassGaussians:
            # plotting black class covariances
            mean, width, height, angle = Gaussian.shape(std=std)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                          edgecolor='black', facecolor='none', 
                          linewidth=0.5, linestyle='--', 
                          label=covlabel, zorder=6)
            ax.add_patch(ell)
            covlabel=None
    
    # plotting colourful datapoints
    xmeans, ymeans = dataset.values.T
    ax.scatter(xmeans, ymeans, s=1, c=dataset.index, label='Unit means', zorder=3)

    # plotting grey samples
    xsample, ysample = mixture.data.values.T
    ax.scatter(xsample, ysample, s=0.5, linewidths=0, c='grey', label='Individual samples', zorder=2)
    
    handles = []
    labels  = []
    for i, Wishart in enumerate(mixture.ClassWisharts):
        # adding data covariances to 2nd legend 
        width, height, angle = Wishart.shape(std=2)
        ell = Ellipse(xy=(0,0), width=width, height=height, angle=angle, 
                      edgecolor="C"+str(i), facecolor='none', 
                      linewidth=0.5, 
                      label='class '+str(i+1))
        handles.append(ell)
        labels.append('Class ' +str(i))
        # ax.add_artist(ell)

    
    # storing 1st legend 
    leg1 = ax.legend(handler_map={Ellipse: HandlerEllipse()}, handletextpad=1,
                     loc='upper right', title="Hierarchical Structure",
                     facecolor='white', scatterpoints=4, framealpha=1)    
    
    leg1.legendHandles[2]._sizes = [2.5]
    
    # adding legends
    leg2 = ax.legend(handles, labels, handler_map={Ellipse: HandlerEllipseRotation()},
                    title="Wishart Scales", loc=("lower left"), ncol=int(np.ceil(mixture.K/2)), facecolor='white',
                    labelspacing=0.7, columnspacing=1, handleheight=1, handlelength=1)
    leg2.get_frame().set_linewidth(.5)
    leg1.get_frame().set_linewidth(.5)
    ax.add_artist(leg1)
    
    # add title
    ax.set_aspect('equal')
    ax.set_title(mixture.info)

    return fig

def plotGaussian(Gaussian, size=20, STDS=[1,2,3], color='black', lw=1, r=1, ax=None):
    if not ax:
        ax = plt.gca()
        
    for i in STDS:
        mean, width, height, angle = Gaussian.shape(std=i)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor=color, facecolor='none', 
                      linewidth=lw, linestyle='--')
        ax.add_patch(ell)
        
    if size:
        samples = Gaussian.samples(size, seed=13)
        x,y = samples.T
        ax.scatter(x,y, color=color, s=r)
        
    return ell

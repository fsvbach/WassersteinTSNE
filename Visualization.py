#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:04:54 2021

@author: fsvbach
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow, Rectangle
from matplotlib.legend_handler import HandlerPatch
import matplotlib.font_manager

inches_per_pt = 1 / 72.27  # convert from pt to inches by multiplying with this number
beamer_paperwidth  = 455.24408 * inches_per_pt
beamer_paperheight = 256.0748 * inches_per_pt
beamer_textwidth   = 398.3386 * inches_per_pt
beamer_textheight  = (beamer_paperheight / inches_per_pt - 40) * inches_per_pt
beamer_contentheight  = (beamer_paperheight / inches_per_pt - 80) * inches_per_pt
ecml_textwidth = 347.12354 * inches_per_pt

imprsgr  = (155 / 255, 155 / 255, 155 / 255) 
mpgreen = (0 / 255, 108 / 255, 102 / 255)
imprsdb = (35 / 255, 127 / 255, 154 / 255) 
TUred   = (141 / 255, 45 / 255, 57 / 255) 
TUgold  = (174 / 255, 159 / 255, 109 / 255)
TUdark  = (55 / 255, 65 / 255, 74 / 255) 
TUcolors = {0: TUred, 1:imprsdb, 2:mpgreen, 3:TUgold, 4:imprsgr, 5:TUdark}

blues = [np.array(imprsdb)*(1-a)+a for a in np.linspace(0,1,128)]
reds  = [np.array(TUred)*a+(1-a)  for a in np.linspace(0,1,128)]
newcolors = np.vstack([blues,reds])
cmap = mpl.colors.ListedColormap(newcolors, name='TUcoolwarm')

beamer = {
    "figure.figsize": (beamer_textwidth, beamer_textheight),
    "text.usetex": True, # <- don't use LaTeX to typeset. It's much slower, and you can't change the font atm.
    # "font.serif": ["Times New Roman"], ######## <- NeurIPS
    # "mathtext.fontset": 'custom',
    # "mathtext.rm" : 'Times New Roman',
    # "mathtext.it" : "Times New Roman:italic",
    # "mathtext.bf" : 'Times New Roman:bold', #####
    # "font.serif": ["Times"], # <- ICML
    #"mathtext.fontset": 'stix', # free ptmx replacement, for ICML and NeurIPS
    # "font.serif": ["CMU Bright"], <- JMLR, if you have Computer Modern on your machine
    # "mathtext.fontset": 'cm', # for JMLR
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif', 
    "axes.labelsize": 7, 
    "font.size": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    'lines.linewidth': 1,
    "axes.facecolor":'none',
    'axes.titlesize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    "text.color": TUdark,
    "axes.edgecolor": TUdark,
    "axes.labelcolor": TUdark,
    "xtick.color": TUdark,
    "ytick.color": TUdark,
}


ecmlpaper = {
    "figure.figsize": (ecml_textwidth, .35*ecml_textwidth),
    "figure.dpi": 300,
    # "text.usetex": True,                
    # "font.family": "sans-serif",
    # 'mathtext.fontset': 'cm',
    # "text.latex.preamble" : r"\usepackage{cmbright}",
    # 'font.sans-serif': ['Computer Modern'],
    # # "text.usetex": False, # <- don't use LaTeX to typeset. It's much slower, and you can't change the font atm.
    # "pgf.texsystem": "pdflatex",
    "axes.labelsize": 7, 
    "font.size": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    'lines.linewidth': 1,
    "axes.facecolor":'none',
    'axes.titlesize': 7,
    'axes.titlepad' : 1,    'axes.linewidth': 0.5}

plt.rcParams.update(ecmlpaper)

def embedScatter(embedding, title, size=1, ax=None):
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
        
def plotMixture(mixture, std=1, ax=None):
    fig = None
    if not ax:
        fig, ax = plt.subplots()
    
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

    # storing 1st legend 
    leg1 = ax.legend(handler_map={Ellipse: HandlerEllipse()}, handletextpad=1,
                     loc='upper right', title="Hierarchical Structure",
                     facecolor='white', scatterpoints=4, framealpha=1)    
    
    leg1.legendHandles[2]._sizes = [2.5]
    
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

def plotWasserstein(Uhist, Vhist, D, opt_res):
    n,m = len(Uhist), len(Vhist)
    emd = round( opt_res.fun,3)
    gamma = opt_res.x.reshape((n, m))

    fig, axes = plt.subplots(2,3, figsize=(22,11*m/n ), 
                             gridspec_kw={'width_ratios': (2*n/10,n,n),
                                          'height_ratios': (m/10,m)})
    [ax.set_axis_off() for ax in axes.ravel()]
    
    axes[0,1].bar(np.arange(n), Uhist, color='C0', alpha=0.5)
    axes[0,1].set(xlim=(-0.5,n-0.5))
    axes[1,0].barh(np.arange(m), Vhist, color='C1', alpha=0.5)
    axes[1,0].set(ylim=(-0.5,m-0.5))
    axes[1,0].invert_xaxis()
    axes[1,0].invert_yaxis()
    
    axes[1,1].imshow(gamma.T, cmap='Greys', vmin=0)
    axes[1,2].imshow(D.T, cmap='Greys', vmin=0)
    
    # axes[0,2].text(0.5,0.5, f"scipy.linprog EMD={emd}", ha='center', fontsize=50)
    fig.tight_layout()
    return fig

def demofigure():
    fig = plt.figure()
    
    fig.patch.set_facecolor('blue')
    fig.patch.set_alpha(0.7)
    
    ax = fig.add_subplot(111)
    
    ax.plot(range(10))
    ax.set(xlabel=f"computer modern: {int(plt.rcParams['axes.labelsize'])}pt")
    ax.patch.set_facecolor('red')
    ax.patch.set_alpha(0.5)

    ax.set_aspect('equal')
    fig.tight_layout()
    
    props = dict(facecolor='white', edgecolor='black')
    fig.text(.5,.5, '$\lambda=0.5$\ncomputer modern: 9pt', ha='center', fontsize=9, bbox=props)
    fig.text(-.2,0.9, 'A', weight='bold', fontsize=9, transform=ax.transAxes, bbox=props)

    # If we don't specify the edgecolor and facecolor for the figure when
    # saving with savefig, it will override the value we set earlier!
    fig.savefig("Plots/DemoFigure.pdf", facecolor=fig.get_facecolor(), edgecolor='none')

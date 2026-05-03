#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 3})
sns.set_style('ticks',
              {'grid.linestyle': 'none', 'axes.edgecolor': '0',
               'axes.linewidth': 1.2, 'legend.frameon': True,
               'xtick.direction': 'out', 'ytick.direction': 'out',
               'xtick.top': True, 'ytick.right': True,
              })

from scipy.constants import m_p, c, e

gamma = 1 + 10**np.linspace(-5, 1, 1000)

def beta(gamma):
    return np.sqrt(1 - gamma**-2)

def plot_Efield_vs_Ekin(gamma=gamma):
    plt.plot(gamma - 1, beta(gamma))
    plt.xscale('log')
    plt.xlabel(r'$\frac{E_{kin}}{m_0c^2} = \gamma-1$')
    plt.ylabel(r'$\beta=v/c$')
    plt.twinx()
    plt.plot(gamma - 1, 1e-6 * beta(gamma) * c * 1)
    plt.ylabel('$E$ [MV/m]');


def plot_quadfield():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a in ax: a.set_aspect('equal')
    
    xlist = np.linspace(-5.0, 5.0, 80)
    ylist = np.linspace(-5.0, 5.0, 80)
    X, Y = np.meshgrid(xlist, ylist)
    Zn = X * Y
    Zs = (X * X - Y * Y) / 2
    levels = np.arange(-6, 6.1, 2)
    
    plt.sca(ax[0])
    plt.contour(X, Y, Zn, levels)
    plt.quiver(X[::5,::5], Y[::5,::5], Y[::5,::5], X[::5,::5])
    plt.title('Normal Quadrupole', y=1.04)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.sca(ax[1])
    plt.contour(X, Y, Zs, levels)
    plt.quiver(X[::5,::5], Y[::5,::5], X[::5,::5], -Y[::5,::5])
    plt.title('Skew Quadrupole', y=1.04)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.tight_layout()


def plot_sextfield():
    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal')
    xlist = np.linspace(-5.0, 5.0, 80)
    ylist = np.linspace(-5.0, 5.0, 80)
    X, Y = np.meshgrid(xlist, ylist)
    Zn = (3 * X**2 * Y - Y**3) / 3
    levels = np.arange(-6, 6.1, 2) * 3
    
    plt.title('Normal Sextupole', y=1.04)
    plt.contour(X, Y, Zn, levels)
    plt.quiver(X[::5,::5], Y[::5,::5], (2 * X * Y)[::5,::5], (X * X - Y * Y)[::5,::5])
    plt.xlabel('$x$')
    plt.ylabel('$y$');


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

import sys
from cpymad.madx import Madx

from scipy.interpolate import interp1d

import PyNAFF

def M_drift(L):
    return np.array([
        [1, L],
        [0, 1]
    ])

def M_dip_x(L, rho0):
    return np.array([
        [np.cos(L / rho0), rho0 * np.sin(L / rho0)],
        [-1 / rho0 * np.sin(L / rho0), np.cos(L / rho0)]
    ])

def M_dip_y(L, rho0):
    return M_drift(L)

def M_quad_x(L, k):
    ksq = np.sqrt(k + 0j)
    return np.array([
        [np.cos(ksq * L), 1 / ksq * np.sin(ksq * L)],
        [-ksq * np.sin(ksq * L), np.cos(ksq * L)]
    ]).real

def M_quad_y(L, k):
    ksq = np.sqrt(k + 0j)
    return np.array([
        [np.cosh(ksq * L), 1 / ksq * np.sinh(ksq * L)],
        [ksq * np.sinh(ksq * L), np.cosh(ksq * L)]
    ]).real

def track(M, u, up):
    '''Apply M to each individual [u;up] vectors value.'''
    return np.einsum('ij,...j->i...', M, np.vstack((u, up)).T)

def track_sext_4D(x, xp, y, yp, mL):
    xp += 0.5 * mL * (y * y - x * x)
    yp += mL * x * y
    return x, xp, y, yp

def plot_twiss(twiss, what='beamline'):
    plt.plot(twiss['s'], twiss['betx'], label=r'$\beta_x$ [m]')
    plt.plot(twiss['s'], twiss['bety'], label=r'$\beta_y$ [m]')
    plt.plot(twiss['s'], twiss['alfx'], label=r'$\alpha_x$ [1]', c='C0', ls='--')
    plt.plot(twiss['s'], twiss['alfy'], label=r'$\alpha_y$ [1]', c='C1', ls='--')
    
    ylim = plt.ylim()
    if what == 'beamline':
        plt.fill_betweenx(ylim, 3-0.3, 3+0.3, color='red', alpha=0.2)
        plt.fill_betweenx(ylim, 7-0.2, 7+0.2, color='red', alpha=0.2)
        plt.fill_betweenx(ylim, 5-0.3, 5+0.3, color='black', alpha=0.2)
    elif what == 'lhc':
        plt.fill_betweenx(ylim, 0, 3.3/2, color='red', alpha=0.2)
        plt.fill_betweenx(ylim, 110/2 - 3.3/2, 110/2 + 3.3/2, color='blue', alpha=0.2)
        plt.fill_betweenx(ylim, 110 - 3.3/2, 110, color='red', alpha=0.2)
    plt.ylim(ylim)
    
    plt.xlabel('$s$ [m]')
    plt.ylabel(r'$\beta_{x,y}$ and $\alpha_{x,y}$')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1));


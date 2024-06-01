# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 14:17:12 2022

@author: dirge
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d')

# ------------------------------------------------------------------

def deathTermFunction(d,y):
# inputs:
# - d equals fixed death term
# - y equals list of densities between 0 and 1    
# output:
# - left side of equil-density equation (d-1)y=(1-y)*(1-e^(-by))
    
    f = [ (d-1)*y[ii] for ii in range(len(y))]
    
    return f

# ------------------------------------------------------------------

def birthTermFunction(b,y):
# inputs:
# - b equals fixed birth term
# - y equals list of densities between 0 and 1    
# output:
# - right side of equil-density equation (d-1)y=(1-y)*(1-e^(-by))    

    f = [ (1-y[ii])*(1-np.exp(-b*y[ii])) for ii in range(len(y))] 
    
    return f

# ------------------------------------------------------------------

def birthTermFunctionInf(y):
# inputs:
# - b equals fixed birth term
# - y equals list of densities between 0 and 1    
# output:
# - right side of equil-density equation (d-1)y=(1-y)*(1-e^(-by))    

    f = [ (1-y[ii]) for ii in range(len(y))] 
    
    return f

# ------------------------------------------------------------------

b1 = 0.1
b2 = 0.8
b3 = 10

d1 = 1.2

y = [ ii/500.0 for ii in range(0,500)]

fd1 = deathTermFunction(d1,y)

fb1  = birthTermFunction(b1,y)
fb2  = birthTermFunction(b2,y)
fb3  = birthTermFunction(b3,y)
fbInf = birthTermFunctionInf(y)

# Figure for Appendix
fig, ax = plt.subplots(1,1)

ax.plot(y,fb1,c='b',linestyle = '-',label = r'RHS: $b=0.1$')
ax.plot(y,fb2,c='g',linestyle = '-',label = r'RHS: $b=0.8$')
ax.plot(y,fb3,c='m',linestyle = '-',label = r'RHS: $b=10$')
ax.plot(y,fbInf,c='k',linestyle = '--',label = r'RHS: $b=\infty$')
ax.plot(y,fd1,c='k',linestyle = '-',label = r'LHS: $d = 1.2$')

ax.set_xticks([ii/10.0 for ii in range(0,11)])
ax.set_xticklabels([str(ii/10.0) for ii in range(0,11)])
# ax.set_xticklabels(["" for ii in range(0,11)])
ax.set_xlim([0,1])

ax.set_yticks([ii*0.5 for ii in range(0,4)])
ax.set_yticklabels([str(ii*0.5) for ii in range(0,4)])
# ax.set_yticklabels(["" for ii in range(0,4)])
ax.set_ylim([0,1.0])

ax.set_xlabel(r'Population density ($\gamma$)')
ax.legend()

fig.savefig(os.getcwd() + '/figures/Appendix/fig_bEvo_DRE_Appendix_quasiEquilibriumDensity.pdf')

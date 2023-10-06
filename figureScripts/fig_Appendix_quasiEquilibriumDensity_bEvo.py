# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 14:17:12 2022

@author: dirge
"""

import numpy as np
import matplotlib.pyplot as plt

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

b1 = 0.5
b2 = 100

d1 = 1.1

y = [ ii/500.0 for ii in range(0,500)]

fd1 = deathTermFunction(d1,y)

fb1  = birthTermFunction(b1,y)
fb2  = birthTermFunction(b2,y)

# Figure for Appendix
fig, ax = plt.subplots(1,1)

ax.plot(y,fb1,c='k',linestyle = '-.',label = r'RHS: $b=0.5$')
ax.plot(y,fb2,c='k',linestyle = '--',label = r'RHS: $b=100$')
ax.plot(y,fd1,c='k',linestyle = '-',label = r'LHS: $d_H = 1.1$')

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

fig.savefig('figures/Appendix/fig_Appendix_quasiEquilibriumDensity_bEvo.pdf')

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: dirge
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d')
from evoLibraries import evo_library as myfun            # my functions in a seperate file

# --------------------------------------------------------------------------
# Calculate Markov Chain Evolution Parameters - Panel (A)
# --------------------------------------------------------------------------
# load parameters from a csv file. The parameters must be provided in the 
# following order (numeric values). All input csv files must be available in 
# inputs or outputs directory. Below are examples of parameters that should be 
# given in the csv file.
#
# T   = 1e9          # carrying capacity
# b   = 2.0          # birth rate
# do  = 100/98.0     # minimum death rate / death rate of optimal genotype
# alpha  = 0.99      # selection coefficient of beneficial mutation in 
# Ua  = 1e-5         # beneficial mutation rate in trait "d"
# Uad = 1e-5         # deleterious mutation rate in trait "d"
# cr  = 0.175        # increment to "c" is (1+cr)
# Ur  = 1e-5         # beneficial mutation rate in trait "c"
# Urd = 1e-5         # deleterious mutation rate in trait "c"
# R  = 1/130.0       # rate of environmental change per iteration
#

# The parameter file is read and a dictionary with their values is generated.
workDir = 'D:/Documents/GitHub/compEvo2d/'
paramFile = workDir + 'inputs/evoExp_DRE_02_parameters.csv'
params = myfun.read_parameterFileDRE(paramFile)

# Calculate absolute fitness state space. This requires specificying:
# dMax  - max size of death term that permits non-negative growth in abundances
# di    - complete of death terms of the various absolute fitness states
[dMax,di,iMax] = myfun.get_absoluteFitnessClassesDRE(params['b'],params['dOpt'],params['alpha'])

# set root solving option for equilibrium densities
# (1) low-density analytic approximation 
# (2) high-density analytic approximation
# (3) root solving numerical approximation
yi_option = 3  

# calculate the state space
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParametersDRE(params,di,iMax,yi_option)

# set pFix values to selection coefficient
pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])

# calculate the evolutionary parameters of the states space
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                    myfun.get_MChainEvoParametersDRE(params,di,iMax,pFixAbs_i,pFixRel_i,yi_option)

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])

ax1.plot(state_i,ve_i,color="black",linewidth=3,label=r'$v_e$')
ax1.scatter(state_i,va_i,color="blue",s=8,label=r'$v_a$')
ax1.scatter(state_i,vr_i,color="red",s=8,label=r'$v_r$')

# axes and label adjustements
iMax = 100
ax1.set_xlim(0,iMax)
ax1.set_ylim(0,5.0e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iMax/100+1)
ax1.set_xticks([100*i for i in range(0,xTickMax)])
ax1.set_xticklabels(["" for i in range(0,xTickMax)],fontsize=16)

# ax1.set_yticks([1e-5*i for i in range(0,5)])
# ax1.set_yticklabels([str(1.0*i) for i in range(0,5)],fontsize=16)

#ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax1.legend(fontsize = 14,ncol=1,loc='lower right')

# annotations
ax1.plot([285,285],[0,1.9e-5],c="black",linewidth=2,linestyle='--')
ax1.annotate("", xy=(280,1.2e-5), xytext=(240,1.2e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
ax1.annotate("", xy=(290,1.2e-5), xytext=(330,1.2e-5),arrowprops={'arrowstyle':'-|>','lw':4})
##plt.text(-84,3.29e-4,r'$i^*=88$',fontsize = 18)
##plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
##plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (B)
# --------------------------------------------------------------------------

paramFile = workDir + 'inputs/evoExp_DRE_01_parameters.csv'
params = myfun.read_parameterFileDRE(paramFile)

[dMax,di,iMax] = myfun.get_absoluteFitnessClassesDRE(params['b'],params['dOpt'],params['alpha'])

# calculate the state space
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParametersDRE(params,di,iMax,yi_option)

# set pFix values to selection coefficient
pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])

# calculate the evolutionary parameters of the states space
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                    myfun.get_MChainEvoParametersDRE(params,di,iMax,pFixAbs_i,pFixRel_i,yi_option)

                    
# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
ax2.plot(state_i,ve_i,color="black",linewidth=3,label=r'$v_e$')
ax2.scatter(state_i,va_i,color="blue",s=8,label=r'$v_a$')
ax2.scatter(state_i,vr_i,color="red",s=8,label=r'$v_r$')

# axes and label adjustements
iMax = 100
ax2.set_xlim(0,iMax)
ax2.set_ylim(0,5.0e-4)       # 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iMax/100+1)
ax2.set_xticks([100*i for i in range(0,xTickMax)])
ax2.set_xticklabels([str(100*i) for i in range(0,xTickMax)],fontsize=16)

# ax2.set_yticks([1e-5*i for i in range(0,5)])
# ax2.set_yticklabels([str(1.0*i) for i in range(0,5)],fontsize=16)

ax2.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax2.legend(fontsize = 14,ncol=1,loc='lower right')

## annotations
ax2.plot([235,235],[0,2.2e-5],c="black",linewidth=2,linestyle='--')
ax2.annotate("", xy=(215,0.7e-5), xytext=(190,0.7e-5),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax2.annotate("", xy=(230,0.5e-5), xytext=(190,0.5e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
##plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
##plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
##plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
# plt.text(15,7.95e-5,r'(A)', fontsize = 22)
# plt.text(15,3.7e-5,r'(B)', fontsize = 22)

# save figure
plt.tight_layout()
fig1.savefig(workDir + 'figures/MainDoc/fig_MarkovChain_VaVeVrIntersectionDRE.pdf')
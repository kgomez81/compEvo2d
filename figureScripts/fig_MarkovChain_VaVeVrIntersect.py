# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
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
# sa  = 1e-2         # selection coefficient of beneficial mutation in 
# Ua  = 1e-5         # max beneficial mutation rate in trait "d"
# Uad = 1e-5         # deleterious mutation rate in trait "d"
# cr  = 0.175        # increment to "c" is (1+cr)
# Ur  = 1e-5         # beneficial mutation rate in trait "c"
# Urd = 1e-5         # deleterious mutation rate in trait "c"
# R  = 1/130.0      # rate of environmental change per iteration
#

# The parameter file is read and a dictionary with their values is generated.
workDir = 'D:/Documents/GitHub/compEvo2d/'
paramFile = workDir + 'inputs/evoExp_RM_01_parameters.csv'
params = myfun.read_parameterFile(paramFile)

# Calculate absolute fitness state space. This requires specificying:
# dMax  - max size of death term that permits non-negative growth in abundances
# di    - complete of death terms of the various absolute fitness states
# iExt  - extinction class
[dMax,di,iExt] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

# pFix values from simulations are loaded for abs-fit mutations to states 0,...,iMax-1 
pFixAbs_File = workDir + 'outputs/evoExp01_absPfix.csv'
pFixAbs_i     = myfun.read_pFixOutputs(pFixAbs_File,iExt)

# pFix values from simulations are loaded for rel-fit mutations at states 1,...,iMax 
pFixRel_File = workDir + 'outputs/evoExp01_relPfix.csv'
pFixRel_i    = myfun.read_pFixOutputs(pFixRel_File,iExt)

# set root solving option for equilibrium densities
# (1) low-density analytic approximation 
# (2) high-density analytic approximation
# (3) root solving numerical approximation
yi_option = 3  

[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                    myfun.get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option)

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])

ax1.plot(state_i,ve_i,color="black",linewidth=3,label=r'$v_e$')
ax1.scatter(state_i,va_i,color="blue",s=8,label=r'$v_a$')
ax1.scatter(state_i,vr_i,color="red",s=8,label=r'$v_r$')

# axes and label adjustements
ax1.set_xlim(-iExt-1,0)
ax1.set_ylim(0,2.52e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iExt/25+1)
ax1.set_xticks([-25*i for i in range(0,xTickMax)])
ax1.set_xticklabels([str(25*i) for i in range(0,xTickMax)],fontsize=16)

#ax1.set_xticklabels(["" for i in range(0,iExt/25+1)],fontsize=16)
ax1.set_yticks([1e-5*5*i for i in range(0,6)])
#ax1.set_yticklabels(["" for i in range(0,6)],fontsize=16)
ax1.set_yticklabels([str(5*i/10.0) for i in range(0,6)],fontsize=16)
#ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax1.legend(fontsize = 14,ncol=1,loc='lower right')

# annotations
ax1.plot([-88,-88],[0,1.6e-4],c="black",linewidth=2,linestyle='--')
ax1.annotate("", xy=(-89,0.7e-4), xytext=(-104,0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
ax1.annotate("", xy=(-87,0.7e-4), xytext=(-72, 0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4})
#plt.text(-84,3.29e-4,r'$i^*=88$',fontsize = 18)
#plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
# plt.text(-175,5.15e-4,r'(A)', fontsize = 22)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (B)
# --------------------------------------------------------------------------

paramFile = workDir + 'inputs/evoExp_RM_02_parameters.csv'
params = myfun.read_parameterFile(paramFile)

[dMax,di,iExt] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

pFixAbs_File = workDir + 'outputs/evoExp01_absPfix.csv'
pFixAbs_i     = myfun.read_pFixOutputs(pFixAbs_File,iExt)

pFixRel_File = workDir + 'outputs/evoExp01_relPfix.csv'
pFixRel_i    = myfun.read_pFixOutputs(pFixRel_File,iExt)

# Calculate all Evo parameters for Markov Chain
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                    myfun.get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option)
                    
# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
ax2.plot(state_i,ve_i,color="black",linewidth=3,label=r'$v_e$')
ax2.scatter(state_i,va_i,color="blue",s=8,label=r'$v_a$')
ax2.scatter(state_i,vr_i,color="red",s=8,label=r'$v_r$')

# axes and label adjustements
ax2.set_xlim(-iExt-1,0)
ax2.set_ylim(0,2.52e-4)       # 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iExt/25+1)
ax2.set_xticks([-25*i for i in range(0,xTickMax)])
ax2.set_xticklabels([str(25*i) for i in range(0,xTickMax)],fontsize=16)

ax2.set_yticks([1e-5*5*i for i in range(0,6)])
ax2.set_yticklabels([str(5*i/10.0) for i in range(0,6)],fontsize=16)
#ax2.set_yticklabels(["" for i in range(0,6)],fontsize=16)
ax2.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax2.legend(fontsize = 14,ncol=1,loc='lower right')

# annotations
ax2.plot([-84,-84],[0,1.52e-4],c="black",linewidth=2,linestyle='--')
ax2.annotate("", xy=(-89,0.8e-4), xytext=(-99,0.8e-4),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax2.annotate("", xy=(-84,0.7e-4), xytext=(-99,0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
#plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
#plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
# plt.text(-175,2.34e-4,r'(B)', fontsize = 22)

# save figure
plt.tight_layout()
fig1.savefig(workDir + 'figures/MainDoc/fig_MarkovChain_VaVeVrIntersection.pdf')
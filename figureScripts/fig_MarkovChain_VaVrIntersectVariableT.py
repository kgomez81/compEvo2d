# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:39:02 2022

@author: dirge
"""


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

import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM

from matplotlib.lines import Line2D

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
paramFilePath = os.getcwd()+'/inputs/evoExp_RM_03_parameters.csv'
# paramFilePath = os.getcwd()+'/inputs/evoExp_RM_04_parameters.csv'
modelType = 'RM'

mcParams = evoObj.evoOptions(paramFilePath, modelType)

T_vals = [1e11,1e7]

mcModels = []

for ii in range(len(T_vals)):
    mcParams.params['T'] = T_vals[ii]
    mcModels = mcModels + [ mcRM.mcEvoModel_RM(mcParams.params) ]

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, ax1 = plt.subplots(1,1,figsize=[7,6])

#ax1.plot(state1_i,ve1_i,color="black",linewidth=2,label=r'$v_e$')
#ax1.plot(eq_y1i,va1_i,color="green",linewidth=2,label=r'$v_{a,1}$')
#ax1.plot(eq_y1i,vr1_i,'-.',color="green",linewidth=2,label=r'$v_{r,1}$')

ax1.plot(mcModels[0].eq_yi,mcModels[0].vd_i,color="blue",linewidth=2,label=r'$v_a (T=10^11)$')
ax1.plot(mcModels[0].eq_yi,mcModels[0].vc_i,'-.',color="blue",linewidth=2,label=r'$v_r (T=10^11)$')

ax1.plot(mcModels[1].eq_yi,mcModels[1].vd_i,color="cyan",linewidth=2,label=r'$v_a (T=10^7)$')
ax1.plot(mcModels[1].eq_yi,mcModels[1].vc_i,'-.',color="cyan",linewidth=2,label=r'$v_r (T=10^7)$')

ax1.scatter([0.9426],[6.718e-4],marker="o",s=40,c="black")
ax1.scatter([0.9496],[3.237e-4],marker="o",s=40,c="black")
ax1.plot([0.9426,0.9496],[6.718e-4,3.237e-4],c="black",linewidth=2,linestyle='-')

# axes and label adjustements
ax1.set_xlim(0,1)
#ax1.set_xticks([-25*i for i in range(0,iExt/25+1)])
#ax1.set_xticklabels([str(25*i) for i in range(0,iExt/25+1)],fontsize=16)
#ax1.set_xticklabels(["" for i in range(0,iExt/25+1)],fontsize=16)
ax1.set_yticks([0.2e-4*i for i in range(0,7)])
ax1.set_yticklabels([str(2*i/10.0) for i in range(0,7)],fontsize=16)
ax1.set_ylim(0,1.3e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])
ax1.set_xlabel(r'Equilibrium Population Density',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)

custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='cyan', lw=2)]
ax1.legend(custom_lines,[r'$T=10^7$',r'$T=10^5$'],fontsize=14)

# save figure
plt.tight_layout()
fig1.savefig('figures/fig_MChain_VaVrIntersectionVariableT.pdf')
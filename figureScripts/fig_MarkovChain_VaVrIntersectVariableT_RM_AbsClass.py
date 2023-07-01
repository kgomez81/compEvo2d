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
import numpy as np

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

T_vals = [1e5,1e7,1e10]
myColors = ["blue","lime","cyan"]
T_vals_strVd = [r'$v_d (T=10^5)$',r'$v_d (T=10^7)$',r'$v_d (T=10^{10})$']
T_vals_strVc = [r'$v_c (T=10^5)$',r'$v_c (T=10^7)$',r'$v_c (T=10^{10})$']
T_vals_strLgd = [r'$T=10^5$',r'$T=10^7$',r'$T=10^{10}$']

mcModels = []

for ii in [0,2]:
    mcParams.params['T'] = T_vals[ii]
    mcModels = mcModels + [ mcRM.mcEvoModel_RM(mcParams.params) ]


# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, ax1 = plt.subplots(1,1,figsize=[7,6])

#ax1.plot(state1_i,ve1_i,color="black",linewidth=2,label=r'$v_e$')
#ax1.plot(eq_y1i,va1_i,color="green",linewidth=2,label=r'$v_{a,1}$')
#ax1.plot(eq_y1i,vr1_i,'-.',color="green",linewidth=2,label=r'$v_{r,1}$')
scaleFactor = 1e3

for ii in range(len(mcModels)):
    ax1.plot(mcModels[ii].state_i,mcModels[ii].vd_i*scaleFactor,color=myColors[ii],linewidth=2,label=T_vals_strVd[ii])
    ax1.plot(mcModels[ii].state_i,mcModels[ii].vc_i*scaleFactor,'-.',color=myColors[ii],linewidth=2,label=T_vals_strVc[ii])

# equilibrium states and equilibrium rates of adaptation
idx = [mcModels[ii].get_mc_stable_state_idx() for ii in range(len(mcModels))]
iEq = np.asarray([mcModels[ii].state_i[idx[ii]] for ii in range(len(mcModels))])
vEq = np.asarray([mcModels[ii].vd_i[idx[ii]] for ii in range(len(mcModels))])

for ii in range(len(mcModels)):
    ax1.scatter(iEq[ii],vEq[ii]*scaleFactor,marker="o",s=40,c="black")

for ii in range(len(mcModels)):
    ax1.plot([iEq[ii],iEq[ii]],[0,vEq[ii]*scaleFactor],c="black",linewidth=1,linestyle=':')

# axes and label adjustements
# ax1.set_xlim(0,1)
xTickMax = int(mcModels[0].get_iExt()/25+1)

ax1.set_xticks([-25*i for i in range(0,xTickMax)])
ax1.set_xticklabels([str(25*i) for i in range(0,xTickMax)],fontsize=16)
ax1.set_yticks([0.2*i for i in range(0,7)])
ax1.set_yticklabels([str(2*i/10.0) for i in range(0,7)],fontsize=16)
ax1.set_ylim(0,1.3)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])
ax1.set_xlabel(r'Absolute Fitness Class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)

arrwLngth1 = 15
arrwOffset = 1
vScale = 0.1
ax1.annotate("", xy=(iEq[0]-arrwLngth1-arrwOffset,vScale*vEq[ii]*scaleFactor), xytext=(iEq[0]-arrwOffset,vScale*vEq[ii]*scaleFactor),arrowprops={'arrowstyle':'->','lw':2})

custom_lines = [Line2D([0], [0], color=myColors[ii], lw=2) for ii in range(len(mcModels))]
ax1.legend(custom_lines,T_vals_strLgd,fontsize=14)

# save figure
plt.tight_layout()
fig1.savefig('figures/MainDoc/fig_MChain_VaVrIntersectionVariableT_AbsClass.pdf')
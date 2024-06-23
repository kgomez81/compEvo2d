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
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

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
# b   = 2.0          # eq_yirth rate
# do  = 100/98.0     # minimum death rate / death rate of optimal genotype
# sa  = 1e-2         # selection coefficient of beneficial mutation in 
# Ua  = 1e-5         # max beneficial mutation rate in trait "d"
# Uad = 1e-5         # deleterious mutation rate in trait "d"
# cr  = 0.175        # increment to "c" is (1+cr)
# Ur  = 1e-5         # beneficial mutation rate in trait "c"
# Urd = 1e-5         # deleterious mutation rate in trait "c"
# R  = 1/130.0      # rate of environmental change per iteration
#

# generate axes for  plotting 
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFilePath1 = os.getcwd()+'/inputs/evoExp_DRE_bEvo_03_parameters.csv'
modelType = 'DRE'
absFitType = 'bEvo'

mcParams = evoObj.evoOptions(paramFilePath1, modelType, absFitType)

T_vals          = [1e7,1e12]
myColors        = ["blue","red"]
myLineStyles    = ['-','-.']
T_vals_strVd    = [r'$v_d (T=10^7)$',r'$v_d (T=10^{12})$']
T_vals_strVc    = [r'$v_c (T=10^7)$',r'$v_c (T=10^{12})$']
T_vals_strLgd   = [r'$T=10^7$',r'$T=10^{12}$']
T_select        = [0,1]
mcModels        = []

for ii in T_select:
    mcParams.params['T'] = T_vals[ii]
    mcModels = mcModels + [ mcDRE.mcEvoModel_DRE(mcParams) ]
    
# some basic paramters for plotting and annotations
scaleFactor     = 1e2
arrwLngth1      = 0.006  
arrwOffset      = 0
vScale          = 0.1
xlow            = 0.76
xhigh           = 0.8
vMaxFact        = 1.5
bOffset         = [0.00,0.00]

for ii in range(len(mcModels)):
    ax1.plot(mcModels[ii].eq_yi,mcModels[ii].va_i*scaleFactor,myLineStyles[ii],color=myColors[0],linewidth=2,label=T_vals_strVd[ii])
    ax1.plot(mcModels[ii].eq_yi,mcModels[ii].vc_i*scaleFactor,myLineStyles[ii],color=myColors[1],linewidth=2,label=T_vals_strVc[ii])

# equilibrium states and equilibrium rates of adaptation
idx = [mcModels[ii].get_mc_stable_state_idx()-1 for ii in range(len(mcModels))]
bEq = np.asarray([mcModels[ii].eq_yi[idx[ii]]+bOffset[ii] for ii in range(len(mcModels))])
vEq = np.asarray([mcModels[ii].va_i[idx[ii]] for ii in range(len(mcModels))])

for ii in range(len(mcModels)):
    ax1.scatter(bEq[ii],vEq[ii]*scaleFactor,marker="o",s=40,c="black")
    
        
for ii in range(len(mcModels)):
    ax1.plot([bEq[ii],bEq[ii]],[0,vEq[ii]*scaleFactor],c="black",linewidth=1,linestyle=':')


xTickVals = [i/100 for i in range(int(xlow*100),int(xhigh*100+1))]
xTickLbls = [str(i/100) for i in range(int(xlow*100),int(xhigh*100+1))]
yTickVals = [np.round(0.2*ii,1) for ii in range(7)]
yTickLbls = [str(np.round(0.2*ii,1)) for ii in range(7)]

# axes and label adjustements
ax1.set_xticks(xTickVals)
ax1.set_xticklabels(xTickLbls,fontsize=16)
ax1.set_xlim(xlow,xhigh)    

# no yticks needed
ax1.set_yticks(yTickVals)
ax1.set_yticklabels(yTickLbls,fontsize=16)
ax1.set_ylim(0,vMaxFact*max(vEq)*scaleFactor)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)

ax1.annotate("", xy=(bEq[0]-arrwLngth1-arrwOffset,vScale*vEq[ii]*scaleFactor), xytext=(bEq[0]-arrwOffset,vScale*vEq[ii]*scaleFactor),arrowprops={'arrowstyle':'->','lw':4})
ax1.text(xhigh-0.1*(xhigh-xlow),vMaxFact*.95*max(vEq)*scaleFactor,r'(A)', fontsize = 22)            

# custom legend
custom_lines = [Line2D([0], [0], linestyle=myLineStyles[ii], color='black', lw=2) for ii in range(len(mcModels))]
ax1.legend(custom_lines,[ T_vals_strLgd[ii] for ii in T_select],fontsize = 20)


# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------

paramFilePath1 = os.getcwd()+'/inputs/evoExp_DRE_bEvo_04_parameters.csv'
modelType = 'DRE'
absFitType = 'bEvo'

mcParams = evoObj.evoOptions(paramFilePath1, modelType, absFitType)

T_vals          = [1e7,1e12]
myColors        = ["blue","red"]
myLineStyles    = ['-','-.']
T_vals_strVd    = [r'$v_d (T=10^7)$',r'$v_d (T=10^{12})$']
T_vals_strVc    = [r'$v_c (T=10^7)$',r'$v_c (T=10^{12})$']
T_vals_strLgd   = [r'$T=10^7$',r'$T=10^{12}$']
T_select        = [0,1]
mcModels        = []

for ii in T_select:
    mcParams.params['T'] = T_vals[ii]
    mcModels = mcModels + [ mcDRE.mcEvoModel_DRE(mcParams) ]
    
# some basic paramters for plotting and annotations
scaleFactor     = 1e6
arrwLngth1      = 0.0129
arrwOffset      = 0
vScale          = 0.1
xlow            = 0.54
xhigh           = 0.60
vMaxFact        = 1.6

for ii in range(len(mcModels)):
    ax2.plot(mcModels[ii].eq_yi,mcModels[ii].va_i*scaleFactor,myLineStyles[ii],color=myColors[0],linewidth=2,label=T_vals_strVd[ii])
    ax2.plot(mcModels[ii].eq_yi,mcModels[ii].vc_i*scaleFactor,myLineStyles[ii],color=myColors[1],linewidth=2,label=T_vals_strVc[ii])

# equilibrium states and equilibrium rates of adaptation
idx = [mcModels[ii].get_mc_stable_state_idx()-1 for ii in range(len(mcModels))]
bEq = np.asarray([mcModels[ii].eq_yi[idx[ii]] for ii in range(len(mcModels))])
vEq = np.asarray([mcModels[ii].va_i[idx[ii]] for ii in range(len(mcModels))])

for ii in range(len(mcModels)):
    ax2.scatter(bEq[ii],vEq[ii]*scaleFactor,marker="o",s=40,c="black")

for ii in range(len(mcModels)):
    ax2.plot([bEq[ii],bEq[ii]],[0,vEq[ii]*scaleFactor],c="black",linewidth=1,linestyle=':')

xTickVals = [i/100 for i in range(int(xlow*100),int(xhigh*100+1))]
xTickLbls = [str(i/100) for i in range(int(xlow*100),int(xhigh*100+1))]
yTickVals = [np.round(0.1*ii,1) for ii in range(6)]
yTickLbls = [str(np.round(0.1*ii,1)) for ii in range(6)]

# axes and label adjustements
ax2.set_xticks(xTickVals)
ax2.set_xticklabels(xTickLbls,fontsize=16)
ax2.set_xlim(xlow,xhigh)

# no yticks needed
ax2.set_yticks(yTickVals)
ax2.set_yticklabels(yTickLbls,fontsize=16)
ax2.set_ylim(0,vMaxFact*max(vEq)*scaleFactor)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

ax2.set_xlabel(r'Population Density ($\gamma$)',fontsize=20,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)

# annotations
ax2.annotate("", xy=(bEq[0]+arrwLngth1-arrwOffset,vScale*vEq[ii]*scaleFactor), xytext=(bEq[0]-arrwOffset,vScale*vEq[ii]*scaleFactor),arrowprops={'arrowstyle':'->','lw':4})
ax2.text(xhigh-0.1*(xhigh-xlow),vMaxFact*.95*max(vEq)*scaleFactor,r'(B)', fontsize = 22)            

# save figure
plt.tight_layout()
fig1.savefig('figures/MainDoc/fig_bEvo_DRE_Decr_Incr_AbsFit_varyT_yAxis.pdf')
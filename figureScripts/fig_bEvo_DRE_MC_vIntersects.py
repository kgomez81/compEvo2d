# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: dirge
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

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
paramFilePath1 = os.getcwd()+'/inputs/evoExp_DRE_bEvo_01_parameters.csv'
modelType = 'DRE'
absFitType = 'bEvo'

mcParams1 = evoObj.evoOptions(paramFilePath1, modelType, absFitType)
mcModel1 = mcDRE.mcEvoModel_DRE(mcParams1)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (B)
# --------------------------------------------------------------------------

paramFilePath2 = os.getcwd()+'/inputs/evoExp_DRE_bEvo_02_parameters.csv'
modelType = 'DRE'
absFitType = 'bEvo'

mcParams2 = evoObj.evoOptions(paramFilePath2, modelType, absFitType)
mcModel2 = mcDRE.mcEvoModel_DRE(mcParams2)

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])

ax1.plot(   mcModel1.state_i, \
            mcModel1.ve_i,color="black",linewidth=3,label=r'$v_e$')
ax1.scatter(mcModel1.state_i, \
            mcModel1.va_i,color="blue",s=8,label=r'$v_b$')
ax1.scatter(mcModel1.state_i, \
            mcModel1.vc_i,color="red",s=8,label=r'$v_c$')

# axes and label adjustements
iMax = mcModel1.get_iMax()
ax1.set_xlim(0,iMax)
ax1.set_ylim(0,1.5e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iMax/50+1)
ax1.set_xticks([50*i for i in range(0,xTickMax)])
ax1.set_xticklabels(["" for i in range(0,xTickMax)],fontsize=16)

ax1.set_yticks([5e-5*i for i in range(0,4)])
ax1.set_yticklabels([str(5*i/10.0) for i in range(0,4)],fontsize=16)

#ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
# ax1.legend(fontsize = 20,ncol=1,loc='lower right')

# annotations
iEq1 = 71
vEq1 = 0.75e-4
arrwLngth1 = 30
ax1.plot([iEq1,iEq1],[0,vEq1],c="black",linewidth=2,linestyle='--')
ax1.annotate("", xy=(iEq1,5.0e-5), xytext=(iEq1-arrwLngth1,5.0e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
ax1.annotate("", xy=(iEq1,5.0e-5), xytext=(iEq1+arrwLngth1,5.0e-5),arrowprops={'arrowstyle':'-|>','lw':4})
#plt.text(iEq1,3.29e-4,r'$x^*=71$',fontsize = 18)
#plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
ax1.text(8,1.39e-4,r'(A)', fontsize = 22)            

# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
ax2.plot(   mcModel2.state_i, \
            mcModel2.ve_i,color="black",linewidth=3,label=r'$v_e$')
ax2.scatter(mcModel2.state_i, \
            mcModel2.va_i,color="blue",s=8,label=r'$v_b$')
ax2.scatter(mcModel2.state_i, \
            mcModel2.vc_i,color="red",s=8,label=r'$v_c$')

# axes and label adjustements
iMax = mcModel2.get_iMax()
ax2.set_xlim(0,iMax)
ax2.set_ylim(0,1.5e-4)       # 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iMax/50+1)
ax2.set_xticks([50*i for i in range(0,xTickMax)])
ax2.set_xticklabels([str(50*i) for i in range(0,xTickMax)],fontsize=16)

ax2.set_yticks([5e-5*i for i in range(0,4)])
ax2.set_yticklabels([str(5*i/10.0) for i in range(0,4)],fontsize=16)

ax2.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax2.legend(fontsize = 20,ncol=1,loc='right')

## annotations
iEq2 = 87
vEq2 = 4.6e-5
arrwLngth2 = 30
ax2.plot([iEq2,iEq2],[0,vEq2],c="black",linewidth=2,linestyle='--')
ax2.annotate("", xy=(iEq2,3.3e-5), xytext=(iEq2-0.6*arrwLngth2,3.3e-5),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax2.annotate("", xy=(iEq2,2.5e-5), xytext=(iEq2-arrwLngth2,2.5e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
#plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
#plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
ax2.text(8,1.39e-4,r'(B)', fontsize = 22)

# diEqStr1 = "%.3f" % (mcModel1.di[iEq1])
# plt.text(120,1.2e-4,'d1*='+diEqStr1,fontsize = 11)
# diEqStr2 = "%.3f" % (mcModel2.di[iEq1])
# plt.text(120,1.0e-4,'d2*='+diEqStr2,fontsize = 11)

# save figure
plt.tight_layout()
fig1.savefig(os.getcwd() + '/figures/MainDoc/fig_MarkovChain_VaVeVrIntersectionDRE_bEvo.pdf')
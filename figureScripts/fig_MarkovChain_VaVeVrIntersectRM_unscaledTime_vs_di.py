# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 21:55:52 2022

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
paramFilePath1 = os.getcwd()+'/inputs/evoExp_RM_01_parameters.csv'
modelType = 'RM'

mcParams1 = evoObj.evoOptions(paramFilePath1, modelType)
mcModel1 = mcRM.mcEvoModel_RM(mcParams1.params)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (B)
# --------------------------------------------------------------------------

paramFilePath2 = os.getcwd()+'/inputs/evoExp_RM_02_parameters.csv'
modelType = 'RM'

mcParams2 = evoObj.evoOptions(paramFilePath2, modelType)
mcModel2 = mcRM.mcEvoModel_RM(mcParams2.params)

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])
ax1.plot(      -mcModel1.di, \
                mcModel1.get_ve_i_perUnitTime()   , color="black",linewidth=3,label=r'$v_e$')
ax1.scatter(   -mcModel1.di, \
                mcModel1.get_vd_i_perUnitTime(),    color="blue",s=8,label=r'$v_d$')
ax1.scatter(   -mcModel1.di, \
                mcModel1.get_vc_i_perUnitTime(),    color="red",s=8,label=r'$v_r$')

# axes and label adjustements
# NOTE: axis order reversed to make graph easier to read/compare
xLb = -(mcModel1.params['b']+1)
xUb = -mcModel1.params['dOpt']
xCnt = int((xUb-xLb)/0.5)+1

ax1.set_xlim(xLb,xUb)
ax1.set_ylim(0,1.0e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

ax1.set_xticks([xLb+0.5*i for i in range(0,xCnt)])
ax1.set_xticklabels(["" for i in range(0,xCnt)],fontsize=16)
ax1.set_yticks([1e-5*2*i for i in range(0,6)])
ax1.set_yticklabels(["%i" % (2*i) for i in range(0,6)],fontsize=16)

#ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax1.legend(fontsize = 14,ncol=1,loc='upper right')

# annotations
iEq1 = 90
vEq1 = 2.00e-5
arrwLngth1 = 16
diEq1 = -mcModel1.di[iEq1]

ax1.plot([diEq1,diEq1],[0,vEq1],c="black",linewidth=2,linestyle='--')
# ax1.annotate("", xy=(-iEq1,0.6e-5), xytext=(-(iEq1 + arrwLngth1),0.6e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
# ax1.annotate("", xy=(-iEq1,0.6e-5), xytext=(-(iEq1 - arrwLngth1),0.6e-5),arrowprops={'arrowstyle':'-|>','lw':4})
#plt.text(-84,3.29e-4,r'$i^*=88$',fontsize = 18)
#plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
ax1.text(xLb+0.02,0.93e-4,r'(A)', fontsize = 22)


# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
ax2.plot(   -mcModel2.di, 
             mcModel2.get_ve_i_perUnitTime(),  color="black",linewidth=3,label=r'$v_e$')
ax2.scatter(-mcModel2.di, \
             mcModel2.get_vd_i_perUnitTime(),  color="blue",s=8,label=r'$v_d$')
ax2.scatter(-mcModel2.di, \
             mcModel2.get_vc_i_perUnitTime(),color="red",s=8,label=r'$v_c$')

# axes and label adjustements
# NOTE: axis order reversed to make graph easier to read/compare
xLb = -(mcModel2.params['b']+1)
xUb = -mcModel2.params['dOpt']
xCnt = int((xUb-xLb)/0.5)+1

ax2.set_xlim(xLb,xUb)
ax2.set_ylim(0,1.0e-4)       # 1.5*max([max(va_i),max(vr_i)])

ax2.set_xticks([xLb+0.5*i for i in range(0,xCnt)])
ax2.set_xticklabels([str(-(xLb+0.5*i)) for i in range(0,xCnt)],fontsize=16)

ax2.set_yticks([1e-5*2*i for i in range(0,6)])
ax2.set_yticklabels(["%i" % (2*i) for i in range(0,6)],fontsize=16)
#ax2.set_yticklabels(["" for i in range(0,6)],fontsize=16)
ax2.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax2.legend(fontsize = 14,ncol=1,loc='upper right')

# annotations
iEq2 = 84
vEq2 = 1.70e-5
arrwLngth2 = 16
diEq2 = -mcModel2.di[iEq2]

ax2.plot([diEq2,diEq2],[0,vEq2],c="black",linewidth=2,linestyle='--')
# ax2.annotate("", xy=(-(iEq2+5),1.0e-5), xytext=(-(iEq2+arrwLngth2),1.0e-5),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
# ax2.annotate("", xy=(-iEq2,0.7e-5), xytext=(-(iEq2+arrwLngth2),0.7e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
#plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
#plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
ax2.text(xLb+0.02,0.93e-4,r'(B)', fontsize = 22)

# diEqStr1 = "%.3f" % (mcModel1.di[iEq1])
# plt.text(-75,0.3e-4,'d1*='+diEqStr1,fontsize = 11)
# diEqStr2 = "%.3f" % (mcModel2.di[iEq2])
# plt.text(-75,0.10e-4,'d2*='+diEqStr2,fontsize = 11)

# save figure
plt.tight_layout()
fig1.savefig(os.getcwd() + '/figures/Supplement/fig_MarkovChain_VaVeVrIntersection_unscaledTime_vs_di.pdf')
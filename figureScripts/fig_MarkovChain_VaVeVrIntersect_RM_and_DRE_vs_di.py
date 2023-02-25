# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:53:54 2023

@author: dirge
"""

import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

# --------------------------------------------------------------------------
# Calculate Markov Chain Evolution Parameters - Panel (A)
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFilePath11 = os.getcwd()+'/inputs/evoExp_RM_01_parameters.csv'
mcParams11 = evoObj.evoOptions(paramFilePath11, 'RM')
mcModel11 = mcRM.mcEvoModel_RM(mcParams11.params)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (B)
# --------------------------------------------------------------------------

paramFilePath12 = os.getcwd()+'/inputs/evoExp_RM_02_parameters.csv'
mcParams12 = evoObj.evoOptions(paramFilePath12, 'RM')
mcModel12 = mcRM.mcEvoModel_RM(mcParams12.params)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (C)
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFilePath21 = os.getcwd()+'/inputs/evoExp_DRE_01_parameters.csv'
mcParams21 = evoObj.evoOptions(paramFilePath21, 'DRE')
mcModel21 = mcDRE.mcEvoModel_DRE(mcParams21.params)

# --------------------------------------------------------------------------
# Recalculate Markov Chain Evolution Parameters - Panel (D)
# --------------------------------------------------------------------------

paramFilePath22 = os.getcwd()+'/inputs/evoExp_DRE_02_parameters.csv'
mcParams22 = evoObj.evoOptions(paramFilePath22, 'DRE')
mcModel22 = mcDRE.mcEvoModel_DRE(mcParams22.params)


# setup figure
fig1, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=[12,12])

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
ax11.plot(      -mcModel11.di, \
                mcModel11.ve_i   , color="black",linewidth=3,label=r'$v_e$')
ax11.scatter(   -mcModel11.di, \
                mcModel11.vd_i,    color="blue",s=8,label=r'$v_d$')
ax11.scatter(   -mcModel11.di, \
                mcModel11.vc_i,    color="red",s=8,label=r'$v_r$')

# axes and label adjustements
# NOTE: axis order reversed to make graph easier to read/compare
xLb = -(mcModel11.params['b']+1)
xUb = -mcModel11.params['dOpt']
xCnt = int((xUb-xLb)/0.5)+1

ax11.set_xlim(xLb,xUb)
ax11.set_ylim(0,2.25e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

ax11.set_xticks([xLb+0.5*i for i in range(0,xCnt)])
# ax11.set_xticklabels([str(-(xLb+0.5*i)) for i in range(0,xCnt)],fontsize=16)
ax11.set_xticklabels(["" for i in range(0,xCnt)],fontsize=16)
ax11.set_yticks([1e-5*5*i for i in range(0,5)])
ax11.set_yticklabels([str(5*i/10.0) for i in range(0,5)],fontsize=16)

ax11.set_ylabel(r'Rate of adaptation (RM)',fontsize=20,labelpad=8)
ax11.legend(fontsize = 14,ncol=1,loc='center left')

# annotations
iEq11 = 90
vEq11 = 1.50e-4
arrwLngth11 = 16
diEq11 = -mcModel11.di[iEq11]

ax11.plot([diEq11,diEq11],[0,vEq11],c="black",linewidth=2,linestyle='--')
# ax1.annotate("", xy=(-iEq,0.6e-4), xytext=(-(iEq + arrwLngth),0.6e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
# ax1.annotate("", xy=(-iEq,0.6e-4), xytext=(-(iEq - arrwLngth),0.6e-4),arrowprops={'arrowstyle':'-|>','lw':4})
#plt.text(-84,3.29e-4,r'$i^*=88$',fontsize = 18)
#plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
ax11.text(xLb+0.02,2.1e-4,r'(A)', fontsize = 22)


# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
ax12.plot(    -mcModel12.di, 
               mcModel12.ve_i,  color="black",linewidth=3,label=r'$v_e$')
ax12.scatter( -mcModel12.di, \
               mcModel12.vd_i,  color="blue",s=8,label=r'$v_d$')
ax12.scatter( -mcModel12.di, \
               mcModel12.vc_i,color="red",s=8,label=r'$v_c$')

# axes and label adjustements
# NOTE: axis order reversed to make graph easier to read/compare
xLb = -(mcModel12.params['b']+1)
xUb = -mcModel12.params['dOpt']
xCnt = int((xUb-xLb)/0.5)+1

ax12.set_xlim(xLb,xUb)
ax12.set_ylim(0,2.25e-4)       # 1.5*max([max(va_i),max(vr_i)])

ax12.set_xticks([xLb+0.5*i for i in range(0,xCnt)])
# ax12.set_xticklabels([str(-(xLb+0.5*i)) for i in range(0,xCnt)],fontsize=16)
ax12.set_xticklabels(["" for i in range(0,xCnt)],fontsize=16)
ax12.set_yticks([1e-5*5*i for i in range(0,5)])
# ax12.set_yticklabels([str(5*i/10.0) for i in range(0,5)],fontsize=16)
ax12.set_yticklabels(["" for i in range(0,5)],fontsize=16)

# ax12.set_xlabel(r'Absolute fitness',fontsize=20,labelpad=8)
# ax12.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax12.legend(fontsize = 14,ncol=1,loc='center left')

# annotations
iEq12 = 84
vEq12 = 1.45e-4
arrwLngth12 = 16
diEq12 = -mcModel12.di[iEq12]
ax12.plot([diEq12,diEq12],[0,vEq12],c="black",linewidth=2,linestyle='--')
# ax2.annotate("", xy=(-(iEq2+5),0.8e-4), xytext=(-(iEq2+arrwLngth2),0.8e-4),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
# ax2.annotate("", xy=(-iEq2,0.7e-4), xytext=(-(iEq2+arrwLngth2),0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
#plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
#plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
ax12.text(xLb+0.02,2.1e-4,r'(B)', fontsize = 22)

# diEqStr1 = "%.3f" % (mcModel1.di[iEq])
# plt.text(-75,0.3e-4,'d1*='+diEqStr1,fontsize = 11)
# diEqStr2 = "%.3f" % (mcModel2.di[iEq])
# plt.text(-75,0.10e-4,'d2*='+diEqStr2,fontsize = 11)

# --------------------------------------------------------------------------
#                               Figure - Panel (C)
# --------------------------------------------------------------------------

ax21.plot(  -mcModel21.di, \
             mcModel21.ve_i,color="black",linewidth=3,label=r'$v_e$')
ax21.scatter(-mcModel21.di, \
              mcModel21.vd_i,color="blue",s=8,label=r'$v_d$')
ax21.scatter(-mcModel21.di, \
              mcModel21.vc_i,color="red",s=8,label=r'$v_c$')

# axes and label adjustements
# NOTE: axis order reversed to make graph easier to read/compare
xLb = -(mcModel21.params['b']+1)
xUb = -mcModel21.params['dOpt']
xCnt = int((xUb-xLb)/0.5)+1

ax21.set_xlim(xLb,xUb)
ax21.set_ylim(0,2.25e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

ax21.set_xticks([xLb+0.5*i for i in range(0,xCnt)])
ax21.set_xticklabels([str(-(xLb+0.5*i)) for i in range(0,xCnt)],fontsize=16)
ax21.set_yticks([5e-5*i for i in range(0,5)])
ax21.set_yticklabels([str(5*i/10.0) for i in range(0,5)],fontsize=16)

ax21.set_xlabel(r'Absolute fitness',fontsize=20,labelpad=8)
ax21.set_ylabel(r'Rate of adaptation (DRE)',fontsize=20,labelpad=8)
ax21.legend(fontsize = 14,ncol=1,loc='center left')

# annotations
iEq21 = 71
vEq21 = 0.85e-4
arrwLngth21 = 30
diEq21 = -mcModel21.di[iEq21]
ax21.plot([diEq21,diEq21],[0,vEq21],c="black",linewidth=2,linestyle='--')
# ax1.annotate("", xy=(iEq1,1.3e-5), xytext=(iEq1-arrwLngth1,1.3e-5),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
# ax1.annotate("", xy=(iEq1,1.3e-5), xytext=(iEq1+arrwLngth1,1.3e-5),arrowprops={'arrowstyle':'-|>','lw':4})
##plt.text(-84,3.29e-4,r'$i^*=88$',fontsize = 18)
##plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
##plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
ax21.text(xLb+0.02,2.1e-4,r'(C)', fontsize = 22)
                    
# --------------------------------------------------------------------------
#                               Figure - Panel (D)
# --------------------------------------------------------------------------
ax22.plot(   -mcModel22.di, \
              mcModel22.ve_i,color="black",linewidth=3,label=r'$v_e$')
ax22.scatter(-mcModel22.di, \
              mcModel22.vd_i,color="blue",s=8,label=r'$v_d$')
ax22.scatter(-mcModel22.di, \
              mcModel22.vc_i,color="red",s=8,label=r'$v_r$')

# axes and label adjustements
# NOTE: axis order reversed to make graph easier to read/compare
xLb = -(mcModel22.params['b']+1)
xUb = -mcModel22.params['dOpt']
xCnt = int((xUb-xLb)/0.5)+1

ax22.set_xlim(xLb,xUb)
ax22.set_ylim(0,2.25e-4)       # 1.5*max([max(va_i),max(vr_i)])

ax22.set_xticks([xLb+0.5*i for i in range(0,xCnt)])
ax22.set_xticklabels([str(-(xLb+0.5*i)) for i in range(0,xCnt)],fontsize=16)
ax22.set_yticks([5e-5*i for i in range(0,5)])
# ax22.set_yticklabels([str(5*i/10.0) for i in range(0,5)],fontsize=16)
ax22.set_yticklabels(["" for i in range(0,5)],fontsize=16)


ax22.set_xlabel(r'Absolute fitness',fontsize=20,labelpad=8)
# ax22.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax22.legend(fontsize = 14,ncol=1,loc='center left')

## annotations
iEq22 = 65
vEq22 = 0.87e-4
arrwLngth22 = 30
diEq22 = -mcModel22.di[iEq22]
ax22.plot([diEq22,diEq22],[0,vEq22],c="black",linewidth=2,linestyle='--')
# ax2.annotate("", xy=(iEq2,0.4e-4), xytext=(iEq2-0.6*arrwLngth2,0.4e-4),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
# ax2.annotate("", xy=(iEq2,0.3e-4), xytext=(iEq2-arrwLngth2,0.3e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
##plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
##plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
##plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
ax22.text(xLb+0.02,2.1e-4,r'(D)', fontsize = 22)

# save figure
plt.tight_layout()
fig1.savefig(os.getcwd() + '/figures/Supplement/fig_MarkovChain_VaVeVrIntersection_RM_and_DRE_vs_di.pdf')
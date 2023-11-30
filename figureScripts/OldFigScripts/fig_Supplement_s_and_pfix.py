# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:57:54 2023

@author: Owner
"""

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

# The parameter file is read and a dictionary with their values is generated.
paramFilePath1 = os.getcwd()+'/inputs/evoExp_RM_01_parameters.csv'
modelType = 'RM'

mcParams_rm = evoObj.evoOptions(paramFilePath1, modelType)
mcModel_rm = mcRM.mcEvoModel_RM(mcParams_rm.params)

# The parameter file is read and a dictionary with their values is generated.
paramFilePath2 = os.getcwd()+'/inputs/evoExp_DRE_01_parameters.csv'
modelType = 'DRE'

mcParams_dre = evoObj.evoOptions(paramFilePath2, modelType)
mcModel_dre = mcDRE.mcEvoModel_DRE(mcParams_dre.params)

#############################################################################
# Generate Supplement FIGURE
#############################################################################

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
# -------------------------------------------------------------------------
# Panel A of figure s/pfix vs death term (RM)
# -------------------------------------------------------------------------

# rescale_s_pfix = 10**2
rescale_s_pfix = 1

ax1.scatter(-mcModel_rm.di[:-1],mcModel_rm.sd_i[:-1]*rescale_s_pfix, \
                    facecolors='none', edgecolors='blue',s=12,label=r'$s_d$')
ax1.scatter(-mcModel_rm.di[:-1],mcModel_rm.pFix_d_i[:-1]*rescale_s_pfix, \
                    facecolors='none', edgecolors='green',s=12,label=r'$\pi_{fix,d}$')
ax1.scatter(-mcModel_rm.di[:-2],mcModel_rm.sc_i[:-2]*rescale_s_pfix,s=12, \
                    facecolors='none', edgecolors='red',label=r'$s_c$')
ax1.scatter(-mcModel_rm.di[:-2],mcModel_rm.pFix_c_i[:-2]*rescale_s_pfix, \
                    facecolors='none', edgecolors='purple',s=12,label=r'$\pi_{fix,c}$')

# ax1.plot(-mcModel_rm.di[:-1],np.log10(mcModel_rm.sd_i[:-1]),c='blue',linewidth=2,label=r'$s_d$')
# ax1.plot(-mcModel_rm.di[:-1],np.log10(mcModel_rm.pFix_d_i[:-1]),c='blue',linestyle='-.',linewidth=2,label=r'$\pi_{fix,d}$')
# ax1.plot(-mcModel_rm.di[:-2],np.log10(mcModel_rm.sc_i[:-2]),c='red',linewidth=2,label=r'$s_c$')
# ax1.plot(-mcModel_rm.di[:-2],np.log10(mcModel_rm.pFix_c_i[:-2]),c='red',linestyle='-.',linewidth=2,label=r'$\pi_{fix,c}$')

# set axis bounds for plots
yTick_lb = 0
yTick_ub = 1.2 * 1e-1 * rescale_s_pfix  # going to rescale to factors of 10^2

xTick_lb = -(mcModel_rm.params['b']+1)
xTick_up = -mcModel_rm.params['dOpt']

ax1.set_xlim([xTick_lb-0.2,xTick_up])
ax1.set_ylim([yTick_lb,yTick_ub])
ax1.set_ylabel(r'$s$ and $\pi_{fix}$',fontsize=16)

ax1.set_xticks([xTick_lb+i*0.5 for i in range(0,4)])
ax1.set_xticklabels(["" for i in range(0,4)],fontsize=16)

# ax1.set_yticks([yTick_lb+i for i in range(0,5)])                          # log scale ticks
# ax1.set_yticklabels([str(yTick_lb+i) for i in range(0,5)],fontsize=16)    # log scale ticks
ax1.set_yticks(     [ 2*i*1e-2*rescale_s_pfix       for i in range(0,7)])
ax1.set_yticklabels([ "%.2f" % (2*i*1e-2*rescale_s_pfix)  \
                                                    for i in range(0,7)],fontsize=16)

ax1.legend(loc='upper right')
ax1.text(-3.15, 1.1 * 1e-1 * rescale_s_pfix, r'(A)', fontsize = 22)
# ax1.text(-3.2,1.21*10**(-1)*rescale_s_pfix,r'$\times 10^{-2}$', fontsize = 14)

# -------------------------------------------------------------------------
# Panel B of figure s/pfix vs death term (DRE)
# -------------------------------------------------------------------------

# rescale_s_pfix = 10**2
rescale_s_pfix = 1

ax2.scatter(-mcModel_dre.di,mcModel_dre.sd_i*rescale_s_pfix, \
                    facecolors='none', edgecolors='blue',s=12,label=r'$s_d$')
ax2.scatter(-mcModel_dre.di,mcModel_dre.pFix_d_i*rescale_s_pfix, \
                    facecolors='none', edgecolors='green',s=12,label=r'$\pi_{fix,d}$')
ax2.scatter(-mcModel_dre.di[1:],mcModel_dre.sc_i[1:]*rescale_s_pfix,s=12, \
                    facecolors='none', edgecolors='red',label=r'$s_c$')
ax2.scatter(-mcModel_dre.di[1:],mcModel_dre.pFix_c_i[1:]*rescale_s_pfix, \
                    facecolors='none', edgecolors='purple',s=12,label=r'$\pi_{fix,c}$')

# ax2.plot(-mcModel_dre.di,np.log10(mcModel_dre.sd_i),c='blue',linewidth=2,label=r'$s_d$')
# ax2.plot(-mcModel_dre.di,np.log10(mcModel_dre.pFix_d_i),c='blue',linestyle='-.',linewidth=2,label=r'$\pi_{fix,d}$')
# ax2.plot(-mcModel_dre.di[1:],np.log10(mcModel_dre.sc_i[1:]),c='red',linewidth=2,label=r'$s_c$')
# ax2.plot(-mcModel_dre.di[1:],np.log10(mcModel_dre.pFix_c_i[1:]),c='red',linestyle='-.',linewidth=2,label=r'$\pi_{fix,c}$')

# set axis bounds for plots
yTick_lb = 0
yTick_ub = 1.2 * 1e-1 * rescale_s_pfix  # going to rescale to factors of 10^2

xTick_lb = -(mcModel_dre.params['b']+1)
xTick_up = -mcModel_dre.params['dOpt']

ax2.set_xlim([xTick_lb-0.2,xTick_up])
ax2.set_ylim([yTick_lb,yTick_ub])
ax2.set_ylabel(r'$s$ and $\pi_{fix}$',fontsize=16)
ax2.set_xlabel(r'Absolute fitness ($d$)',fontsize=16)

ax2.set_xticks([xTick_lb+i*0.5 for i in range(0,4)])
ax2.set_xticklabels([str(-(xTick_lb+i*0.5)) for i in range(0,4)],fontsize=16)

# ax2.set_yticks([yTick_lb+i for i in range(0,5)])                          # log scale ticks
# ax2.set_yticklabels([str(yTick_lb+i) for i in range(0,5)],fontsize=16)    # log scale ticks
ax2.set_yticks(     [ 2*i*1e-2*rescale_s_pfix       for i in range(0,7)])
ax2.set_yticklabels([ "%.2f" % (2*i*1e-2*rescale_s_pfix)  \
                                                    for i in range(0,7)],fontsize=16)

ax2.legend(loc='upper right')
ax2.text(-3.15, 1.1 * 1e-1 * rescale_s_pfix, r'(B)', fontsize = 22)
# ax2.text(-3.2,1.21*10**(-1)*rescale_s_pfix,r'$\times 10^{-2}$', fontsize = 14)

plt.tight_layout()
fig.savefig(os.getcwd() + '/figures/Supplement/fig_Supplement_s_and_pfix.pdf')
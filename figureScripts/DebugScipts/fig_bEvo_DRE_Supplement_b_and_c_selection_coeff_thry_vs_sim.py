# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: Kevin Gomez
Plot showing Pfix values and selection coefficients

NOTE: These figures rely on the data generated from fig_bEvo_DRE_MC_vIntersections

"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import time
import pickle 

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.MarkovChain import MC_factory as mcFac

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# filepaths for loading and saving outputs
inputsPath  = os.path.join(os.getcwd(),'inputs')
outputsPath = os.path.join(os.getcwd(),'outputs')
figSavePath = os.path.join(os.getcwd(),'figures','Supplement')

# filenames and paths for saving outputs
figFile     = 'fig_bEvo_DRE_pfix_vs_s.pdf'
figDatDir   = 'fig_bEvo_DRE_MC_vInt_pfix1'
paramFile   = ['evoExp_DRE_bEvo_01_parameters.csv','evoExp_DRE_bEvo_02_parameters.csv']
paramTag    = ['param_01_DRE_bEvo','param_02_DRE_bEvo']
saveDatFile = [''.join(('_'.join((figDatDir,pTag)),'.pickle')) for pTag in paramTag]

# set paths to generate output files for tracking progress of loop/parloop
mcModelOutputPath   = os.path.join(outputsPath,figDatDir) 
figFilePath         = os.path.join(figSavePath,figFile)

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# define model list (empty), model type and abs fitness axis.
mcModels    = []
modelType   = 'DRE'
absFitType  = 'bEvo'
nMc         = len(paramFile)

# get the mcArray data
if not (os.path.exists(mcModelOutputPath)):
    # if the data does not exist then generate it
    os.mkdir(mcModelOutputPath)
    
    # start timer
    tic = time.time()
    
    # generate models
    for ii in range(nMc):
        # generate mcModels
        mcModels.append(mcFac.mcFactory().newMcModel( os.path.join(inputsPath,paramFile[ii]), \
                                                   modelType,absFitType))
        # save the data to a pickle file
        with open(os.path.join(mcModelOutputPath,saveDatFile[ii]), 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(mcModels[-1], file)
        
    print(time.time()-tic)

else:
    # load mcModel data
    for ii in range(nMc):
        # if data exist, then just load it to generate the figure
        with open(os.path.join(mcModelOutputPath,saveDatFile[ii]), 'rb') as file:
            # Serialize and write the variable to the file
            mcModels.append(pickle.load(file))
            
#%% ------------------------------------------------------------------------
# generate figures
# --------------------------------------------------------------------------


fig,ax1 = plt.subplots(1,1,figsize=[7,5])
# -------------------------------------------------------------------------
# Panel A of figure s/pfix vs death term (RM)
# -------------------------------------------------------------------------

# rescale_s_pfix = 10**2
rescale_s_pfix = 1.0

ax1.scatter(mcModels[-1].bi[:-1],mcModels[-1].sa_i[:-1]*rescale_s_pfix, \
                    facecolors='none', edgecolors='blue',s=12,label=r'$s_b$')
ax1.scatter(mcModels[-1].bi[:-1],mcModels[-1].pFix_a_i[:-1]*rescale_s_pfix, \
                    facecolors='none', edgecolors='green',s=12,label=r'$\pi_{fix,b}$')
ax1.scatter(mcModels[-1].bi[:-2],mcModels[-1].sc_i[:-2]*rescale_s_pfix,s=12, \
                    facecolors='none', edgecolors='red',label=r'$s_c$')
ax1.scatter(mcModels[-1].bi[:-2],mcModels[-1].pFix_c_i[:-2]*rescale_s_pfix, \
                    facecolors='none', edgecolors='purple',s=12,label=r'$\pi_{fix,c}$')

# ax1.plot(-mcModels[-1].bi[:-1],np.log10(mcModels[-1].sd_i[:-1]),c='blue',linewidth=2,label=r'$s_d$')
# ax1.plot(-mcModels[-1].bi[:-1],np.log10(mcModels[-1].pFix_d_i[:-1]),c='blue',linestyle='-.',linewidth=2,label=r'$\pi_{fix,d}$')
# ax1.plot(-mcModels[-1].bi[:-2],np.log10(mcModels[-1].sc_i[:-2]),c='red',linewidth=2,label=r'$s_c$')
# ax1.plot(-mcModels[-1].bi[:-2],np.log10(mcModels[-1].pFix_c_i[:-2]),c='red',linestyle='-.',linewidth=2,label=r'$\pi_{fix,c}$')

# set axis bounds for plots
yTick_lb = 0
yTick_ub = 1.4*np.max(mcModels[-1].sa_i)*rescale_s_pfix  # going to rescale to factors of 10^2

xTick_lb = mcModels[-1].params['d']-1
xTick_up = mcModels[-1].bi[-1]

# ax1.set_xlim([xTick_lb-0.2,xTick_up])
ax1.set_ylim([yTick_lb,yTick_ub])
ax1.set_ylabel(r'$s$ and $\pi_{fix}$',fontsize=16)
ax1.set_xlabel(r'Absolute fitness ($b$)',fontsize=16)

# ax1.set_xticks([xTick_lb+i*0.5 for i in range(0,4)])
# ax1.set_xticklabels(["" for i in range(0,4)],fontsize=16)

# ax1.set_yticks([yTick_lb+i for i in range(0,5)])                          # log scale ticks
# ax1.set_yticklabels([str(yTick_lb+i) for i in range(0,5)],fontsize=16)    # log scale ticks
# ax1.set_yticks(     [ 2*i*1e-2*rescale_s_pfix       for i in range(0,7)])
# ax1.set_yticklabels([ "%.2f" % (2*i*1e-2*rescale_s_pfix) for i in range(0,7)],fontsize=16)

ax1.legend(loc='upper right')
# ax1.text(-3.15, 1.1 * 1e-1 * rescale_s_pfix, r'(A)', fontsize = 22)
# ax1.text(-3.2,1.21*10**(-1)*rescale_s_pfix,r'$\times 10^{-2}$', fontsize = 14)

plt.tight_layout()

# save figure
fig.savefig(figFilePath,bbox_inches='tight')
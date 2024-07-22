# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: dirge
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt

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
figSavePath = os.path.join(os.getcwd(),'figures','MainDoc')

# filenames and paths for saving outputs
figFile     = 'fig_bEvo_DRE_MC_vIntersections.pdf'
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

fig, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------

ax1.plot(   mcModels[0].state_i, \
            mcModels[0].ve_i,color="black",linewidth=3,label=r'$v_E$')
ax1.scatter(mcModels[0].state_i, \
            mcModels[0].va_i,color="blue",s=8,label=r'$v_b$')
ax1.scatter(mcModels[0].state_i, \
            mcModels[0].vc_i,color="red",s=8,label=r'$v_c$')

# axes and label adjustements
iMax = mcModels[0].get_iMax()
ax1.set_xlim(2,iMax)
ax1.set_ylim(0,0.31e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iMax/50+1)
ax1.set_xticks([50*i for i in range(1,xTickMax)])
ax1.set_xticklabels(["" for i in range(1,xTickMax)],fontsize=16)

ax1.set_yticks([1e-5*i for i in range(0,4)])
ax1.set_yticklabels([str(1*i/10.0) for i in range(0,4)],fontsize=16)

#ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
# ax1.legend(fontsize = 20,ncol=1,loc='lower right')

# annotations
iEq1 = 59
vEq1 = 0.14e-4
arrwLngth1 = 25
ax1.plot([iEq1,iEq1],[0,vEq1],c="black",linewidth=2,linestyle='--')
ax1.annotate("", xy=(iEq1,0.5*vEq1), xytext=(iEq1-arrwLngth1,0.5*vEq1),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
ax1.annotate("", xy=(iEq1,0.5*vEq1), xytext=(iEq1+arrwLngth1,0.5*vEq1),arrowprops={'arrowstyle':'-|>','lw':4})
#plt.text(iEq1,3.29e-4,r'$x^*=71$',fontsize = 18)
#plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
ax1.text(15,0.29e-4,r'(A)', fontsize = 22) 
ax1.legend(fontsize = 20,ncol=1,loc='upper right')           

# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
ax2.plot(   mcModels[1].state_i, \
            mcModels[1].ve_i,color="black",linewidth=3,label=r'$v_E$')
ax2.scatter(mcModels[1].state_i, \
            mcModels[1].va_i,color="blue",s=8,label=r'$v_b$')
ax2.scatter(mcModels[1].state_i, \
            mcModels[1].vc_i,color="red",s=8,label=r'$v_c$')

# axes and label adjustements
iMax = mcModels[1].get_iMax()
ax2.set_xlim(2,iMax)
ax2.set_ylim(0,0.31e-4)       # 1.5*max([max(va_i),max(vr_i)])

xTickMax = int(iMax/50+1)
ax2.set_xticks([50*i for i in range(1,xTickMax)])
ax2.set_xticklabels([str(50*i) for i in range(1,xTickMax)],fontsize=16)

ax2.set_yticks([1e-5*i for i in range(0,4)])
ax2.set_yticklabels([str(1*i/10.0) for i in range(0,4)],fontsize=16)

ax2.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)

## annotations
iEq2 = 106
vEq2 = 0.63e-5
arrwLngth2 = 25
ax2.plot([iEq2,iEq2],[0,vEq2],c="black",linewidth=2,linestyle='--')
ax2.annotate("", xy=(iEq2,0.75*vEq2), xytext=(iEq2-0.6*arrwLngth2,0.75*vEq2),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax2.annotate("", xy=(iEq2,0.5*vEq2), xytext=(iEq2-arrwLngth2,0.5*vEq2),arrowprops={'arrowstyle':'-|>','lw':4,'color':'red'})
#plt.text(-78,0.29e-4,r'$i^*=84$',fontsize = 18)
#plt.text(-78,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
#plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
ax2.text(15,0.29e-4,r'(B)', fontsize = 22)

# diEqStr1 = "%.3f" % (mcModels[0].di[iEq1])
# plt.text(120,1.2e-4,'d1*='+diEqStr1,fontsize = 11)
# diEqStr2 = "%.3f" % (mcModels[1].di[iEq1])
# plt.text(120,1.0e-4,'d2*='+diEqStr2,fontsize = 11)

plt.show()
plt.tight_layout()

# save figure
fig.savefig(figFilePath,bbox_inches='tight')
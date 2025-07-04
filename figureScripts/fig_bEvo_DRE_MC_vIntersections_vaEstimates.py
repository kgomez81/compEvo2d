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
from matplotlib.lines import Line2D

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

# Annotations Parameters
iEq1        = 59
vEq1        = 0.14e-4
vEq1_hgt    = 0.35
arrwLngth1  = 49
arrw_hgt    = 4
arrw_hlngth = 15

xAbs    = iEq1-arrwLngth1-1
yAbs    = vEq1_hgt*vEq1
dxAbs   = arrwLngth1  
dyAbs   = 0
wdthAbs = 0.05* vEq1

xEnv    = iEq1+arrwLngth1+1
yEnv    = vEq1_hgt*vEq1
dxEnv   = -arrwLngth1  
dyEnv   = 0
wdthEnv = 0.05*vEq1

# Annotations
ax1.plot([iEq1,iEq1],[0,vEq1],c="black",linewidth=2,linestyle='--')
ax1.arrow(xAbs, yAbs, dxAbs, dyAbs, length_includes_head=True, \
          width = wdthAbs ,head_width= arrw_hgt*wdthAbs, head_length=arrw_hlngth, color='blue')
ax1.arrow(xEnv, yEnv, dxEnv, dyEnv, length_includes_head=True, \
          width = wdthEnv ,head_width= arrw_hgt*wdthEnv, head_length=arrw_hlngth, color='black')
ax1.text(15,0.29e-4,r'(A)', fontsize = 22) 

# ax1.legend(fontsize = 20,ncol=1,loc='upper right')    

# custom legend

myColors        = ["black","blue","red"]
myLineStyles    = ['-','-','-']
T_vals_strLgd   = [r'$v_E$',r'$v_b$',r'$v_c$']
custom_lines = [Line2D([0], [0], linestyle=myLineStyles[ii], color=myColors[ii], lw=2) for ii in range(len( T_vals_strLgd ))]
ax1.legend(custom_lines,T_vals_strLgd,fontsize = 20)

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

# Annotations Parameters
iEq2        = 106
vEq2        = 0.63e-5
vEq2_hgt_1  = 0.55
vEq2_hgt_2  = 0.20
arrwLngth21 = 0.6*40
arrwLngth22 = 40
arrw_hgt_1  = 4
arrw_hgt_2  = 5
arrw_hlngt1 = 10
arrw_hlngt2 = 15

xAbs    = iEq2-arrwLngth21
yAbs    = vEq2_hgt_1*vEq2
dxAbs   = arrwLngth21 
dyAbs   = 0
wdthAbs = 0.05*vEq2

xRel    = iEq2-arrwLngth22
yRel    = vEq2_hgt_2*vEq2
dxRel   = arrwLngth22  
dyRel   = 0
wdthRel = 0.05*vEq2

# Annotations
ax2.plot([iEq2,iEq2],[0,vEq2],c="black",linewidth=2,linestyle='--')
ax2.arrow(xAbs, yAbs, dxAbs, dyAbs, length_includes_head=True, \
          width = wdthAbs ,head_width= arrw_hgt_1*wdthAbs, head_length=arrw_hlngt1, color='blue')
ax2.arrow(xRel, yRel, dxRel, dyRel, length_includes_head=True, \
          width = wdthRel ,head_width= arrw_hgt_2*wdthRel, head_length=arrw_hlngt2, color='red')
ax2.text(15,0.29e-4,r'(B)', fontsize = 22)

plt.show()
plt.tight_layout()

# save figure
fig.savefig(figFilePath,bbox_inches='tight')
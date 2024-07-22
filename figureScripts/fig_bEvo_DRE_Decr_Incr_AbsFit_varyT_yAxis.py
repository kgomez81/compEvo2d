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
import time
import pickle 

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries import evoObjects as evoObj
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
figFile     = 'fig_bEvo_DRE_Decr_Incr_AbsFit_varyT_yAxis.pdf'
figDatDir   = 'fig_bEvo_DRE_DecrIncrAbsFit_pfix1'
paramFile   = ['evoExp_DRE_bEvo_03_parameters.csv','evoExp_DRE_bEvo_04_parameters.csv']
paramTag    = ['param_01_DRE_bEvo','param_02_DRE_bEvo']
saveDatFile = [''.join(('_'.join((figDatDir,pTag)),'.pickle')) for pTag in paramTag]

# set paths to generate output files for tracking progress of loop/parloop
mcModelOutputPath   = os.path.join(outputsPath,figDatDir) 
figFilePath         = os.path.join(figSavePath,figFile)

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# define model list (empty), model type and abs fitness axis.
mcModels    = [[],[]]
modelType   = 'DRE'
absFitType  = 'bEvo'
nMc         = len(paramFile)

# parameters for generating sets of models
T_vals          = [1e7,1e12]
myColors        = ["blue","red"]
myLineStyles    = ['-','-.']
T_vals_strVd    = [r'$v_d (T=10^7)$',r'$v_d (T=10^{12})$']
T_vals_strVc    = [r'$v_c (T=10^7)$',r'$v_c (T=10^{12})$']
T_vals_strLgd   = [r'$T=10^7$',r'$T=10^{12}$']
T_select        = [0,1]

# get the mcArray data
if not (os.path.exists(mcModelOutputPath)):
    # if the data does not exist then generate it
    os.mkdir(mcModelOutputPath)
    
    # start timer
    tic = time.time()
    
    # generate models
    for ii in range(nMc):

        mcParams = evoObj.evoOptions(os.path.join(inputsPath,paramFile[ii]), \
                                                         modelType, absFitType)
        for tVals in T_vals:
            mcParams.params['T'] = tVals
            mcModels[ii].append( mcFac.mcFactory().createMcModel(mcParams) )
                            
        # save the data to a pickle file
        with open(os.path.join(mcModelOutputPath,saveDatFile[ii]), 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(mcModels[ii], file)
        
    print(time.time()-tic)

else:
    # load mcModel data
    for ii in range(nMc):
        # if data exist, then just load it to generate the figure
        with open(os.path.join(mcModelOutputPath,saveDatFile[ii]), 'rb') as file:
            # Serialize and write the variable to the file
            mcModels[ii].append(pickle.load(file))

#%% ------------------------------------------------------------------------
# generate figures
# --------------------------------------------------------------------------

# generate axes for  plotting 
fig, (ax1,ax2) = plt.subplots(2,1,figsize=[7,12])

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
                    
# some basic paramters for plotting and annotations
scaleFactor     = 1e2
arrwLngth1      = 0.006  
arrwOffset      = 0
vScale          = 0.1
xlow            = 0.76
xhigh           = 0.8
vMaxFact        = 1.5
bOffset         = [0.00,0.00]

for ii in range(len(mcModels[0])):
    ax1.plot(mcModels[0][ii].eq_yi,mcModels[0][ii].va_i*scaleFactor,myLineStyles[ii],color=myColors[0],linewidth=2,label=T_vals_strVd[ii])
    ax1.plot(mcModels[0][ii].eq_yi,mcModels[0][ii].vc_i*scaleFactor,myLineStyles[ii],color=myColors[1],linewidth=2,label=T_vals_strVc[ii])

# equilibrium states and equilibrium rates of adaptation
idx = [mcModels[0][ii].get_mc_stable_state_idx()-1 for ii in range(len(mcModels[0]))]
bEq = np.asarray([mcModels[0][ii].eq_yi[idx[ii]]+bOffset[ii] for ii in range(len(mcModels[0]))])
vEq = np.asarray([mcModels[0][ii].va_i[idx[ii]] for ii in range(len(mcModels[0]))])

for ii in range(len(mcModels[0])):
    ax1.scatter(bEq[ii],vEq[ii]*scaleFactor,marker="o",s=40,c="black")
        
for ii in range(len(mcModels[0])):
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
custom_lines = [Line2D([0], [0], linestyle=myLineStyles[ii], color='black', lw=2) for ii in range(len(mcModels[0]))]
ax1.legend(custom_lines,[ T_vals_strLgd[ii] for ii in T_select],fontsize = 20)


# --------------------------------------------------------------------------
#                               Figure - Panel (B)
# --------------------------------------------------------------------------
    
# some basic paramters for plotting and annotations
scaleFactor     = 1e6
arrwLngth1      = 0.0129
arrwOffset      = 0
vScale          = 0.1
xlow            = 0.54
xhigh           = 0.60
vMaxFact        = 1.6

for ii in range(len(mcModels[1])):
    ax2.plot(mcModels[1][ii].eq_yi,mcModels[1][ii].va_i*scaleFactor,myLineStyles[ii],color=myColors[0],linewidth=2,label=T_vals_strVd[ii])
    ax2.plot(mcModels[1][ii].eq_yi,mcModels[1][ii].vc_i*scaleFactor,myLineStyles[ii],color=myColors[1],linewidth=2,label=T_vals_strVc[ii])

# equilibrium states and equilibrium rates of adaptation
idx = [mcModels[1][ii].get_mc_stable_state_idx()-1 for ii in range(len(mcModels[1]))]
bEq = np.asarray([mcModels[1][ii].eq_yi[idx[ii]] for ii in range(len(mcModels[1]))])
vEq = np.asarray([mcModels[1][ii].va_i[idx[ii]] for ii in range(len(mcModels[1]))])

for ii in range(len(mcModels[1])):
    ax2.scatter(bEq[ii],vEq[ii]*scaleFactor,marker="o",s=40,c="black")

for ii in range(len(mcModels[1])):
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

plt.show()
plt.tight_layout()

# save figure
fig.savefig(figFilePath,bbox_inches='tight')
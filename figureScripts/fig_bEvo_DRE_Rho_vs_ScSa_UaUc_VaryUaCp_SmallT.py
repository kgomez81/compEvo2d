# -*- coding: utf-8 -*-
"""
Created on Sun May 08 11:22:43 2022

@author: dirge
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

from evoLibraries.MarkovChain import MC_array_class as mcArry
import figFunctions as figFun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# filepaths for loading and saving outputs
inputsPath  = os.path.join(os.getcwd(),'inputs')
outputsPath = os.path.join(os.getcwd(),'outputs')
figSavePath = os.path.join(os.getcwd(),'figures','Supplement')

# filenames for saving outputs
figFile     = 'fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp_SmallT.pdf'
figDatDir   = 'fig_bEvo_DRE_RhoUaCpSmallT_pfix2'
paramFile   = 'evoExp_DRE_bEvo_07_parameters.csv'
paramTag    = 'param_07_DRE_bEvo'
saveDatFile = ''.join(('_'.join((figDatDir,paramTag)),'.pickle'))

# set paths to generate output files for tracking progress of loop/parloop
mcArrayOutputPath   = os.path.join(outputsPath,figDatDir) 
saveDatFilePath     = os.path.join(mcArrayOutputPath,saveDatFile)
figFilePath         = os.path.join(figSavePath,figFile)

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.path.join(inputsPath,paramFile)
modelType  = 'DRE'
absFitType = 'bEvo'

# set list of variable names that will be used to specify the grid
# and the bounds with increments needed to define the grid.
# varNames[0] = string with dictionary name of evo model parameter
# varNames[1] = string with dictionary name of evo model parameter
varNames       = ['Ua','cp']

# varBounds values define the min and max bounds of parameters that are used to 
# define the square grid. First index j=0,1 (one for each evo parameter). 
# varBounds[0]    = list of base 10 exponentials to use in forming the parameter 
#                   grid for X1
# varBounds[1]    = list of base 10 exponentials to use in forming the parameter 
#                   grid for X2
# NOTE: both list should include 0 to represent the center points of the grid.
#       For example, [-2,-1,0,1,2] would designate [1E-2,1E-1,1E0,1E1,1e2].
#       Also note that the entries don't have to be integers.
nArry     = 11

Ua_Bnds = np.linspace(-3, 3, nArry)
cp_Bnds = np.linspace(-1, 1, nArry)   # cannot exceed ~O(10^-1) for pFix estimates

varBounds = [Ua_Bnds, cp_Bnds]

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# get the mcArray data
if not (os.path.exists(mcArrayOutputPath)):
    # if the data does not exist then generate it
    os.mkdir(mcArrayOutputPath)
    
    # generate grid
    tic = time.time()
    mcModels = mcArry.mcEvoGrid(paramFilePath, modelType, absFitType, varNames, varBounds, mcArrayOutputPath)
    print(time.time()-tic)
    
    # save the data to a pickle file
    outputs  = [paramFilePath, modelType, absFitType, varNames, varBounds, mcModels]
    with open(saveDatFilePath, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(outputs, file)

else:
    # if data exist, then just load it to generate the figure
    with open(saveDatFilePath, 'rb') as file:
        # Serialize and write the variable to the file
        loaded_data = pickle.load(file)
        
    paramFilePath   = loaded_data[0]
    modelType       = loaded_data[1]
    absFitType      = loaded_data[2]
    varNames        = loaded_data[3]
    varBounds       = loaded_data[4]
    mcModels        = loaded_data[5]

#%% ------------------------------------------------------------------------
# construct plot variables
# --------------------------------------------------------------------------

X = np.log10(mcModels.eff_sc_ij / mcModels.eff_sa_ij)   # sc/sd
Y = np.log10(mcModels.eff_Ua_ij / mcModels.eff_Uc_ij)   # Ud/Uc
Z = np.log10(mcModels.rho_ij)                        # rho

[x,y,z] = figFun.getScatterData(X,Y,Z)

zRange = np.max(np.abs(z))

#%% ------------------------------------------------------------------------
#                           Plot data
# --------------------------------------------------------------------------

# set up a figure 
fig, ax1 = plt.subplots(1,1,figsize=[9,7])

# plot a 3D surface like in the example mplot3d/surface3d_demo
map1 = ax1.scatter(x, y, c=z, s=40, cmap='bwr', vmin = -zRange, vmax = +zRange, edgecolor='none')

ax1.set_xlabel(r'$log_{10}(s_c/s_b)$',fontsize=26,labelpad=8)
ax1.set_ylabel(r'$log_{10}(U_b/U_c)$',fontsize=26,labelpad=8)

xMin = int(np.floor(min(x)))
xMax = int(np.ceil(max(x))+1)

yMin = int(np.floor(min(y)))
yMax = int(np.ceil(max(y))+1)

xTicks      = [-1,-0.5,0,0.5,1]
xTickLbls   = [str(0.1),'',str(1),'',str(10)]

yTicks      = [-3,-2,-1,0,1,2]
yTickLbls   = [str(0.001),str(0.01),str(0.1),str(1),str(10),str(100)]

zIncr    = 0.05
zMaxMod5 = int(np.ceil(zRange/zIncr))
zTicks   = [np.round(zIncr*ii,2) for ii in range(-zMaxMod5, zMaxMod5+1)]
zLabels  = [str(tick) for tick in zTicks]

ax1.set_xticks(xTicks)
ax1.set_xticklabels(xTickLbls,fontsize=22)

ax1.set_yticks(yTicks)
ax1.set_yticklabels(yTickLbls,fontsize=22)

plt.grid(True)

cbar = fig.colorbar(map1, ax=ax1, ticks = zTicks)
cbar.ax.set_yticklabels(zLabels) 
cbar.ax.tick_params(labelsize=18)

plt.show()
plt.tight_layout()

# save figure
fig.savefig(figFilePath,bbox_inches='tight')


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
sys.path.insert(0, os.getcwd() + '..\\')

from evoLibraries.MarkovChain import MC_array_class as mcArry
# from evoLibraries.MarkovChain import MC_functions as mcFun

def getScatterData(X,Y,Z):
    
    x = []
    y = []
    z = []
    
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            
            # removed bad data
            xGood = not np.isnan(X[ii,jj]) and not np.isinf(X[ii,jj])
            yGood = not np.isnan(Y[ii,jj]) and not np.isinf(Y[ii,jj])
            zGood = not np.isnan(Z[ii,jj]) and not np.isinf(Z[ii,jj])
            
            if xGood and yGood and zGood:
                x = x + [ X[ii,jj] ]
                y = y + [ Y[ii,jj] ]
                z = z + [ Z[ii,jj] ]
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    return [x,y,z]

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.getcwd()+'/inputs/evoExp_DRE_bEvo_09_parameters.csv'
modelType = 'DRE'
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

mcArrayOutputPath = os.getcwd() + '\\outputs\\fig_bEvo_DRE_Rho_sb0_vary'

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# generate grid
tic = time.time()
mcModels = mcArry.mcEvoGrid(paramFilePath, modelType, absFitType, varNames, varBounds, mcArrayOutputPath)
print(time.time()-tic)

# save the data to a pickle file
outputs  = [paramFilePath, modelType, absFitType, varNames, varBounds, mcModels]
saveOutputsPath = os.getcwd()+'/outputs/fig_bEvo_DRE_Rho_T_large/fig_bEvo_Rho_small_T_evoExp_DRE_bEvo_09_parameters.pickle'
with open(saveOutputsPath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(outputs, file)

## To load the data, just run the imports section, followed by code below
# saveOutputsPath = os.getcwd()+'/outputs/fig_bEvo_DRE_Rho_vs_eff_ScSa_and_eff_UaUc_evoExp_DRE_bEvo_06_parameters.pickle'

# with open(saveOutputsPath, 'rb') as file:
#     # Serialize and write the variable to the file
#     loaded_data = pickle.load(file)
    
# paramFilePath   = loaded_data[0]
# modelType       = loaded_data[1]
# absFitType      = loaded_data[2]
# varNames        = loaded_data[3]
# varBounds       = loaded_data[4]
# mcModels        = loaded_data[5]

#%% ------------------------------------------------------------------------
# construct plot variables
# --------------------------------------------------------------------------

X = np.log10(mcModels.eff_sc_ij / mcModels.eff_sa_ij)   # sc/sd
Y = np.log10(mcModels.eff_Ua_ij / mcModels.eff_Uc_ij)   # Ud/Uc
Z = mcModels.rho_ij                                     # rho

[x,y,z] = getScatterData(X,Y,Z)

zRange = np.max(np.abs(z-1))

#%% ------------------------------------------------------------------------
#                           Plot data
# --------------------------------------------------------------------------

# set up a figure 
fig, ax1 = plt.subplots(1,1,figsize=[9,7])

# plot a 3D surface like in the example mplot3d/surface3d_demo
map1 = ax1.scatter(x, y, c=z, s=40, cmap='bwr', vmin = 1-zRange, vmax = 1+zRange, edgecolor='none')

ax1.set_xlabel(r'$log_{10}(s_c/s_b)$',fontsize=26,labelpad=8)
ax1.set_ylabel(r'$log_{10}(U_b/U_c)$',fontsize=26,labelpad=8)

xMin = int(np.floor(min(x)))
xMax = int(np.ceil(max(x))+1)

yMin = int(np.floor(min(y)))
yMax = int(np.ceil(max(y))+1)

# ax1.set_xticks([0.5*ii for ii in range(xMin-1,xMax+1)])
# ax1.set_xticklabels([str(0.5*ii) for ii in range(xMin-1,xMax+1)],fontsize=22)

# ax1.set_yticks([ii for ii in range(yMin,yMax)])
# ax1.set_yticklabels([str(ii) for ii in range(yMin,yMax)],fontsize=22)

xTicks      = [-1,-0.5,0,0.5,1]
xTickLbls   = [str(0.1),'',str(1),'',str(10)]

yTicks      = [-3,-2,-1,0,1,2]
yTickLbls   = [str(0.001),str(0.01),str(0.1),str(1),str(10),str(100)]
               
ax1.set_xticks(xTicks)
ax1.set_xticklabels(xTickLbls,fontsize=22)

ax1.set_yticks(yTicks)
ax1.set_yticklabels(yTickLbls,fontsize=22)

plt.grid(True)

cbar = fig.colorbar(map1, ax=ax1)
cbar.ax.tick_params(labelsize=18)


plt.show()
plt.tight_layout()

fig.savefig(os.getcwd() + '/figures/MainDoc/fig_bEvo_DRE_Rho_vs_eff_ScSa_and_eff_UaUc_T_large.pdf',bbox_inches='tight')


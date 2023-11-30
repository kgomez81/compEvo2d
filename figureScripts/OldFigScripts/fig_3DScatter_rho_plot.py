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

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d')

from evoLibraries.MarkovChain import MC_array_class as mcArry
from evoLibraries.MarkovChain import MC_functions as mcFun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.getcwd()+'/inputs/evoExp_RM_06_parameters.csv'
modelType = 'RM'

# set list of variable names that will be used to specify the grid
# and the bounds with increments needed to define the grid.
# varNames[0] = string with dictionary name of evo model parameter
# varNames[1] = string with dictionary name of evo model parameter
varNames       = ['UdMax','cp']

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

UdMax_Bnds = np.linspace(-3, 3, nArry)
cp_Bnds = np.linspace(-1, 1, nArry)   # cannot exceed ~O(10^-1) for pFix estimates

varBounds = [UdMax_Bnds, cp_Bnds]

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# generate grid
tic = time.time()
mcModels = mcArry.mcEvoGrid(paramFilePath, modelType, varNames, varBounds)
print(time.time()-tic)

#%% ------------------------------------------------------------------------
# construct contour plot grids
# --------------------------------------------------------------------------
nGridCt = 51

X = np.log10(mcModels.eff_sc_ij / mcModels.eff_sd_ij)
Y = np.log10(mcModels.eff_Ud_ij / mcModels.eff_Uc_ij)
Z = mcModels.rho_ij

X_noNan = X
Y_noNan = Y

for ii in range(Z.shape[0]):
    for jj in range(Z.shape[1]):
        if np.isnan(Z[ii,jj]):
            X_noNan[ii,jj] = np.nan
            Y_noNan[ii,jj] = np.nan

#%% ------------------------------------------------------------------------
#                           Plot data
# --------------------------------------------------------------------------

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X.flatten(),Y.flatten(),Z.flatten()) 
# ax.set_xlabel('s ratio')
# ax.set_ylabel('U ratio')
# ax.set_zlabel('rho')

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=[7,7])

# =============
# First subplot
# =============
# set up the axes for the first plot
ax1 = fig.add_subplot(1, 1, 1)

# plot a 3D surface like in the example mplot3d/surface3d_demo
ax1.scatter(X_noNan, Y_noNan, c=Z, s=45, cmap='jet')
map1 = ax1.imshow(cmap='jet')

ax1.set_xlabel(r'$log_{10}(s_c/s_d)$',fontsize=26,labelpad=8)
ax1.set_ylabel(r'$log_{10}(U_d/U_c)$',fontsize=26,labelpad=8)

ax1.set_xticks([i/2.0 for i in range(-2,2)])
ax1.set_xticklabels([str(i/2.0) for i in range(-2,2)],fontsize=22)
ax1.set_yticks([ii for ii in range(-4,2)])
ax1.set_yticklabels([str(ii) for ii in range(-4,2)],fontsize=22)

plt.grid()

ax1.text(-0.9,1.3,r'(A)', fontsize = 26)    

# ax1.set_yticks([i for i in range(int(np.floor(np.min(Y))),int(np.ceil(np.max(Y))))])
# ax1.set_yticklabels([str(i) for i in range(int(np.floor(np.min(Y))),int(np.ceil(np.max(Y))))])

# ax.set_zlim(-1.01, 1.01)
# fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# fig.savefig(os.getcwd() + '/figures/MainDoc/fig_3DScatter_rho_plot_PanelA.pdf',bbox_inches='tight')

# ==============
# Second subplot
# ==============

fig = plt.figure(figsize=[7,7],constrained_layout=True)

# set up the axes for the second plot
ax2 = fig.add_subplot(1, 1, 1, projection='3d')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
ax2.scatter(X, Y, Z, s=30)
ax2.zaxis.set_rotate_label(False)
ax2.set_xlabel(r'$log_{10}(s_c/s_d)$',fontsize=16,labelpad=12)
ax2.set_ylabel(r'$log_{10}(U_d/U_c)$',fontsize=16,labelpad=12)
ax2.set_zlabel(r'$\rho$',fontsize=20,labelpad=12,rotation=0)

ax2.set_xticks([i/2.0 for i in range(-2,2)])
ax2.set_xticklabels([str(i/2.0) for i in range(-2,2)],fontsize=16)

ax2.set_yticks([2*ii for ii in range(-2,1)])
ax2.set_yticklabels([str(2*ii) for ii in range(-2,1)],fontsize=16)

ax2.set_zticks([round(0.1*ii,1) for ii in range(7,12)])
ax2.set_zticklabels([str(round(0.1*ii,1)) for ii in range(7,12)],fontsize=16)

ax2.view_init(elev=12, azim=-105)

ax2.text(-1,1.3,1.13,r'(B)', fontsize = 18)    

# ax2.set_yticks([i for i in range(int(np.floor(np.min(Y))),int(np.ceil(np.max(Y))))])
# ax2.set_yticklabels([str(i) for i in range(int(np.floor(np.min(Y))),int(np.ceil(np.max(Y))))])

# plt.tight_layout()

plt.show()

# fig.savefig(os.getcwd() + '/figures/MainDoc/fig_3DScatter_rho_plot_PanelB.pdf',bbox_inches='tight')


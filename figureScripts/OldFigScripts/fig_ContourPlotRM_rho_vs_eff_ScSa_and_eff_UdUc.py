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

[xi, yi, zi] = mcFun.get_contourPlot_arrayData(X, Y, Z, nGridCt)

#%% ------------------------------------------------------------------------
#                           Plot data
# --------------------------------------------------------------------------

# fig, ax1 = plt.subplots(nrows=1)
# ax1.contourf(xi, yi, zi)
# cntr1 = ax1.contourf(xi, yi, zi, cmap="summer")
# fig.colorbar(cntr1, ax=ax1)

# ax1.set( xlim=(xi.min(), xi.max()), ylim=(yi.min(), yi.max()) )
# ax1.set_title(r'$\rho$ Contour Plot')
# ax1.set_xlabel(r'$\log_{10}(s_c/s_d)$')
# ax1.set_ylabel(r'$\log_{10}(U_d/U_c)$')

# plt.show()


fig, ax1 = plt.subplots(nrows=1)

myLvls = np.linspace(np.round(np.min(zi),1),np.round(np.max(zi,1)+0.1,1),15)
myLineLvls = np.asarray([0.5,0.80,0.90,1.0])

# ax1.contourf(xi, yi, zi)
cntr1 = ax1.contourf(xi, yi, zi,level=myLvls,cmap="summer")
fig.colorbar(cntr1, ax=ax1)

cpl = ax1.contour(xi,yi,zi,colors='k',levels = myLineLvls)
ax1.clabel(cpl, fmt='%2.1f', colors='k', fontsize=11)

ax1.set( xlim=(xi.min(), xi.max()), ylim=(yi.min(), yi.max()) )
ax1.set_title(r'$\rho$ Contour Plot')
ax1.set_xlabel(r'$\log_{10}(s_c/s_d)$')
ax1.set_ylabel(r'$\log_{10}(U_d/U_c)$')

plt.show()

fig.savefig(os.getcwd() + '/figures/MainDoc/fig_ContourPlotRM_rho_vs_eff_ScSa_and_eff_UdUc.pdf')


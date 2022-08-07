# -*- coding: utf-8 -*-
"""
Created on Sun May 08 11:22:43 2022

@author: dirge
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import matplotlib.tri as tri

import evo_library as myfun            # evo functions in seperate file

# --------------------------------------------------------------------------
#                               Get data/options
# --------------------------------------------------------------------------

# set up figure parameters and options
paramFilePath   = 'inputs/evoExp_RM_02_parameters.csv'
outputFilePath  = 'outputs/dat_rho_eff_UvsS_evoExp_RM_02.pickle'
saveFigName     = 'figures/fig_rho_eff_UvsS_evoExp_RM_02.pdf'
yi_option       = 3
modelType       = 'RM'

# square array is built with first two parmeters, and second set are held constant.
# varNames[0][0] stored as X1_ARRY
# varNames[1][0] stored as X1_ref
# varNames[0][1] stored as X2_ARRY
# varNames[1][1] stored as X2_ref
varNames        = [['cr','UaMax'],['sa','Ur']]  

# varBounds values define the min and max bounds of parameters that are used to 
# define the square grid. 
# varBounds[j][0] = min Multiple of parameter value in file (Xj variable)
# varBounds[j][1] = max Multiple of parameter value in file (Xj variable)
# varBounds[j][2] = number of increments from min to max (log scale) 
gridCnt         = 41
varBounds       = [[1e-1,1e+1,gridCnt],[1e-1,1e+1,gridCnt]]

myOptions = myfun.evoOptions(paramFilePath,outputFilePath,saveFigName,modelType,varNames,varBounds,yi_option)

# --------------------------------------------------------------------------
#                               generate data
# --------------------------------------------------------------------------
# os.remove(myOptions.saveDataName)

if not os.path.isfile(myOptions.saveDataName):
    myfun.get_contourPlot_arrayData(myOptions)
    
# --------------------------------------------------------------------------
#                               Load data
# --------------------------------------------------------------------------

# load the array data for rho and variable parameter
with open(myOptions.saveDataName, 'rb') as f:
    [X1_ARRY,X2_ARRY,RHO_ARRY,Y_ARRY,X1_ref,X2_ref, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp,dMax] = pickle.load(f)

# Notes
# X1_ARRY  = VARYING relative fitness cr coefficient
# X1_ref   = SINGLE VALUE/FIXED absolute fitness selection coefficient sa
# X2_ARRY  = VARYING absolute fitness max mutation rate UaMax
# X2_ref   = SINGLE VALUE/FIXED relative fitness mutation rate Ur
    
# --------------------------------------------------------------------------
#                           Screen data
# --------------------------------------------------------------------------
# Remove any points with beneficial mutation rates that are near the extinction
# class mutation rate (use tolerance Ua/UaMax -1 = 5%).
    
x = []
y = []
z = []

nPts = 0

for ii in range(X1_ARRY.shape[0]):
    for jj in range(X1_ARRY.shape[1]):
        if (abs(np.log10(effUa_ARRY[ii,jj])-np.log10(X2_ARRY[ii,jj])) >= 2 ):
            x = x + [ np.log10(effSr_ARRY[ii,jj]/X1_ref) ]
            y = y + [ np.log10(effUa_ARRY[ii,jj]/X2_ref) ]
            z = z + [ RHO_ARRY[ii,jj] ]
            nPts = nPts + 1

x = np.reshape(np.asarray(x),nPts)     # convert to log10 space
y = np.reshape(np.asarray(y),nPts)     # convert to log10 space
z = np.reshape(np.asarray(z),nPts)

# --------------------------------------------------------------------------
#                           Interpolate data
# --------------------------------------------------------------------------

# set size of grid used for interpolation
nGridCt = 21 

xMin = np.min(x)
xMax = np.max(x)
yMin = np.min(y)
yMax = np.max(y)

# Create grid values first.
xi = np.linspace(xMin, xMax, nGridCt)
yi = np.linspace(yMin, yMax, nGridCt)

# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# --------------------------------------------------------------------------
#                           Plot data
# --------------------------------------------------------------------------

fig, ax1 = plt.subplots(nrows=1)
ax1.contourf(xi, yi, zi)
cntr1 = ax1.contourf(xi, yi, zi, cmap="summer")
fig.colorbar(cntr1, ax=ax1)
#ax1.plot(x, y, 'ko', ms=3)
ax1.set(xlim=(xMin, xMax), ylim=(yMin,yMax))
ax1.set_title(r'$\rho$ Contour Plot')
ax1.set_xlabel(r'$\log_{10}(s_r/s_a)$')
ax1.set_ylabel(r'$\log_{10}(U_a/U_r)$')
plt.show()
fig.savefig(myOptions.saveFigName)


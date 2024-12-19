# -*- coding: utf-8 -*-
"""
Created on Sun May 08 11:22:43 2022

@author: Kevin Gomez

Script to generate figure 3 of manuscript.
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

import figClasses as fc
import figFunctions as ff

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# filepaths for loading and saving outputs
figPathsIO  = dict()
figPathsIO['saveFigSubdir'] = 'MainDoc'
figPathsIO['figFile']       = 'fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp.pdf'
figPathsIO['figDatDir']     = 'fig_bEvo_DRE_RhoUaCp_pfix1'
figPathsIO['paramFile']     = 'evoExp_DRE_bEvo_06_parameters.csv'
figPathsIO['paramTag']      = 'param_06_DRE_bEvo'

# specify parameters for the MC models
figModelIO = dict()
figModelIO['modelType']     = 'DRE'
figModelIO['absFitType']    = 'bEvo'
figModelIO['varNames']      = ['Ua','cp']
figModelIO['nArry']         = 11

Ua_Bnds = np.linspace(-3, 3, figModelIO['nArry'])
cp_Bnds = np.linspace(-1, 1, figModelIO['nArry'])   # cannot exceed ~O(10^-1) for pFix estimates
figModelIO['varBounds']     = [Ua_Bnds, cp_Bnds]

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

rhoFig  = fc.figRhoPlot(figPathsIO,figModelIO)
figData = rhoFig.get_rhoPlotData()

#%% ------------------------------------------------------------------------
#                           Plot data
# --------------------------------------------------------------------------

# set up a figure 
fig, ax1 = plt.subplots(1,1,figsize=[9,7])

# plot a 3D surface like in the example mplot3d/surface3d_demo
map1 = ax1.scatter(figData['xData'], 
                   figData['yData'], 
                   c=figData['zData'], 
                   s=40, cmap='bwr', 
                   vmin = figData['zBnds'][0], 
                   vmax = figData['zBnds'][1], 
                   edgecolor='none')

# adding markers for points used as basis for figure 2
ax1.scatter(figData['log10_ss'][2,0] , figData['log10_UU'][2,0] ,
            marker='s',s=100,color='cyan',facecolors='none',linewidths=2)

ax1.scatter(figData['log10_ss'][8,10], figData['log10_UU'][8,10] ,
            marker='s',s=100,color='magenta',facecolors='none',linewidths=2)

# set x-axis attributes
ax1.set_xlabel(r'$log_{10}(s_c/s_b)$',fontsize=26,labelpad=8)
ax1.set_xticks(figData['xTick']) 
ax1.set_xticklabels(figData['xLbls'],fontsize=22)

# set y-axis attributes
ax1.set_ylabel(r'$log_{10}(U_b/U_c)$',fontsize=26,labelpad=0)
ax1.set_yticks(figData['yTick'])
ax1.set_yticklabels(figData['yLbls'],fontsize=22)
ax1.set_ylim([-2.1,2.1])

# set z-axis (colormap) attributes
cbar = fig.colorbar(map1, ax=ax1, ticks = figData['zTick'])
cbar.ax.set_yticklabels(figData['zLbls'])
cbar.ax.tick_params(labelsize=18)
cbar.set_label(r'$\rho$',size=26,rotation=0,labelpad=20,y=0.52)

plt.grid(True)
plt.show()
plt.tight_layout()

# save figure
fig.savefig(rhoFig.figFilePath,bbox_inches='tight')


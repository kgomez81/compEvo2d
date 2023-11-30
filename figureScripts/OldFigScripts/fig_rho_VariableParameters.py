# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 03:47:26 2022

@author: dirge
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import evo_library as myfun            # my functions in a seperate file
from mpl_toolkits.mplot3d import Axes3D
import copy as cpy

# --------------------------------------------------------------------------
# get parameters and array of values
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp_RM_02_parameters.csv'
params = myfun.read_parameterFile(paramFile)

# set root solving option for equilibrium densities
# (1) low-density analytic approximation 
# (2) high-density analytic approximation
# (3) root solving numerical approximation
yi_option = 3  

#T_vals = np.logspace(5,15,num=11)

varParam1 = 'UaMax'
P1_vals = np.logspace(-6,-4,num=9)
varParam2 = 'Ur'
P2_vals = np.logspace(-6,-4,num=9)

#varParam1 = 'sa'
#P1_vals = np.logspace(-3,-1,num=9)
#varParam2 = 'cr'
#P2_vals = np.logspace(-3,-1,num=9)

P1_ARRY, P2_ARRY = np.meshgrid(P1_vals, P2_vals)
RHO_ARRY = np.zeros(P1_ARRY.shape)
# d_ARRY = np.zeros(P1_ARRY.shape)
Y_ARRY = np.zeros(P1_ARRY.shape)

paramsTemp = cpy.copy(params)

# --------------------------------------------------------------------------
# Calculated rho values for T vs 2nd parameter variable
# --------------------------------------------------------------------------

for ii in range(int(P1_ARRY.shape[0])):
    for jj in range(int(P2_ARRY.shape[1])):
        
        paramsTemp[varParam1] = P1_ARRY[ii,jj]
        paramsTemp[varParam2] = P2_ARRY[ii,jj]
        
        # Calculate absolute fitness state space. 
        [dMax,di,iExt] = myfun.get_absoluteFitnessClasses(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['sa'])
        
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(paramsTemp,di,iExt,yi_option)        
        
        pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
        pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])
        
        # Use s values for pFix until we get sim pFix values can be obtained
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                        myfun.get_MChainEvoParameters(paramsTemp,di,iExt,pFixAbs_i,pFixRel_i,yi_option)
 
        RHO_ARRY[ii,jj] = myfun.get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i)   
        Y_ARRY[ii,jj] = myfun.get_intersection_popDensity(va_i, vr_i, eq_yi)   


# --------------------------------------------------------------------------
# Contour plot of rho values
# --------------------------------------------------------------------------

myLvls = np.linspace(np.round(np.min(RHO_ARRY),1),np.round(np.max(RHO_ARRY),1)+0.1,40)
myLineLvls = np.asarray([0.5,0.80,0.90,1.0])

fig, ax1 = plt.subplots(1,1,figsize=[7,6])
#cp = ax1.contourf(np.log10(T_ARRY), np.log10(P_ARRY), RHO_ARRY, levels = myLvls)
#cpl = ax1.contour(np.log10(T_ARRY), np.log10(P_ARRY), RHO_ARRY,colors='k',levels = myLineLvls)

cp = ax1.contourf(np.log10(P1_ARRY), np.log10(P2_ARRY), RHO_ARRY, levels = myLvls)
cpl = ax1.contour(np.log10(P1_ARRY), np.log10(P2_ARRY), RHO_ARRY,colors='k',levels = myLineLvls)

ax1.clabel(cpl, fmt='%2.1f', colors='k', fontsize=11)
fig.colorbar(cp) # Add a colorbar to a plot
ax1.set_title(r'$\rho$ Contour Plot')
ax1.set_xlabel('log10 '+varParam1)
ax1.set_ylabel('log10 '+varParam2)
plt.show()

# --------------------------------------------------------------------------

fig1 = plt.figure()
ax1 = plt.subplot(111, projection='3d')
#cp = ax1.plot_surface(np.log10(T_ARRY), np.log10(SA_ARRY), RHO_ARRY)
cp = ax1.scatter(np.log10(T_ARRY), np.log10(P_ARRY), RHO_ARRY)
ax1.view_init(elev=35., azim=110)
ax1.set_xlabel('log10 T')
#ax1.set_ylabel('log10 ' + varParam)
ax1.set_ylabel(varParam)
plt.show()

# --------------------------------------------------------------------------

# Gamma plots
myLvls = np.linspace(np.round(np.min(Y_ARRY),1),1,40)

fig, ax1 = plt.subplots(1,1,figsize=[7,6])
cp = ax1.contourf(np.log10(T_ARRY), np.log10(P_ARRY), Y_ARRY,levels = myLvls)
#cp.set_clim([0,1])
#cp.set_cmap('hsv')
fig.colorbar(cp) # Add a colorbar to a plot
ax1.set_title(r'$\gamma^*$ Contour Plot')
ax1.set_xlabel('log10 T')
ax1.set_ylabel('log10 ' + varParam)
plt.show()

fig1 = plt.figure()
ax1 = plt.subplot(111, projection='3d')
#cp = ax1.plot_surface(np.log10(T_ARRY), np.log10(SA_ARRY), RHO_ARRY)
cp = ax1.scatter(np.log10(T_ARRY), np.log10(P_ARRY), Y_ARRY)
ax1.view_init(elev=20., azim=240)
plt.show()


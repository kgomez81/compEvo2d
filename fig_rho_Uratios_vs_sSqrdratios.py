# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:28:45 2022
@author: kgomez for Masel Lab
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
import pickle 

# --------------------------------------------------------------------------
# get parameters and array of values
# --------------------------------------------------------------------------

def get_contourPlot_arrayData():
    # The parameter file is read and a dictionary with their values is generated.
    paramFile = 'inputs/evoExp_RM_02_parameters.csv'
    params = myfun.read_parameterFile(paramFile)
    
    # set root solving option for equilibrium densities
    # (1) low-density analytic approximation 
    # (2) high-density analytic approximation
    # (3) root solving numerical approximation
    yi_option = 3  
    
    # set U valuses
    varParam1A = 'UaMax'
    varParam1B = 'Ur'
    P1_vals = np.logspace(-6,-4,num=21)
    P1_valMid = np.logspace(-5,-5,num=1)
    
    # set s values
    varParam2A = 'cr'
    varParam2B = 'sa'
    P2_vals = np.logspace(-3,-1,num=21)
    P2_valMid = np.logspace(-2,-2,num=1)
    
    P1_ARRY, P2_ARRY = np.meshgrid(P1_vals, P2_vals)
    RHO_ARRY = np.zeros(P1_ARRY.shape)
    Y_ARRY = np.zeros(P1_ARRY.shape)

    # arrays to store effecttive s and U values
    effSa_ARRY = np.zeros(P1_ARRY.shape)
    effUa_ARRY = np.zeros(P1_ARRY.shape)

    effSr_ARRY = np.zeros(P1_ARRY.shape)
    effUr_ARRY = np.zeros(P1_ARRY.shape)
    
    paramsTemp = cpy.copy(params)

    # --------------------------------------------------------------------------
    # Calculated rho values for T vs 2nd parameter variable
    # --------------------------------------------------------------------------
    
    for ii in range(int(P1_ARRY.shape[0])):
        for jj in range(int(P2_ARRY.shape[1])):
            
            # set Ua values and Ur values
            paramsTemp[varParam1A] = P1_ARRY[ii,jj]
            paramsTemp[varParam1B] = P1_valMid[0]
            
            # set cr and sa values
            paramsTemp[varParam2A] = P2_ARRY[ii,jj]
            paramsTemp[varParam2B] = P2_valMid[0]
            
            # Calculate absolute fitness state space. 
            [dMax,di,iExt] = myfun.get_absoluteFitnessClasses(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['sa'])
            
            [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(paramsTemp,di,iExt,yi_option)        
            
            pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
            pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])
            
            # Use s values for pFix until we get sim pFix values can be obtained
            [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                            myfun.get_MChainEvoParameters(paramsTemp,di,iExt,pFixAbs_i,pFixRel_i,yi_option)
     
            [RHO_ARRY[ii,jj], effSa_ARRY[ii,jj], effUa_ARRY[ii,jj], effSr_ARRY[ii,jj], effUr_ARRY[ii,jj] ] = \
                            myfun.get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i,eq_Ni)   
                            
            Y_ARRY[ii,jj] = myfun.get_intersection_popDensity(va_i, vr_i, eq_yi)   
    
    
    with open('outputs/fig_rho_UvsS_data.pickle', 'wb') as f:
        pickle.dump([P1_ARRY,P2_ARRY,RHO_ARRY,Y_ARRY,P1_valMid,P2_valMid, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp,dMax], f)
    
    return [P1_ARRY,P2_ARRY,RHO_ARRY,Y_ARRY,P1_valMid,P2_valMid, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp,dMax]
    
# --------------------------------------------------------------------------
# Contour plot of rho or gamma values
# --------------------------------------------------------------------------

scriptCases = dict({'generateData': 0,'loadData':1,'getRhoFig':2,'getGammaFig':3,'getEffEvoParamFig':4})

# select what you want the script to do.
plotCase = scriptCases['getEffEvoParamFig']

if (plotCase == 0):
    [P1_ARRY, P2_ARRY, RHO_ARRY, Y_ARRY, P1_valMid, P2_valMid, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp, dMax] = get_contourPlot_arrayData()
    
elif (plotCase == 1):
    # load the array data for rho and variable parameter
    with open('outputs/fig_rho_UvsS_data.pickle', 'rb') as f:
        [P1_ARRY, P2_ARRY, RHO_ARRY, Y_ARRY, P1_valMid, P2_valMid, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp, dMax] = pickle.load(f)
        
if (plotCase > 1):
    
    X1_ARRY = ((1/P1_valMid[0])*P1_ARRY)
    X2_ARRY = ((1/P2_valMid[0])*P2_ARRY)
    
    # generate figure and select which contour plot to create 
    fig, ax1 = plt.subplots(1,1,figsize=[7,6])
    
    if (plotCase == 2):
        myLvls = np.linspace(np.round(np.min(RHO_ARRY),1),np.round(np.max(RHO_ARRY),1)+0.1,40)
        myLineLvls = np.asarray([0.5,0.80,0.90,1.0])
    
        cp = ax1.contourf(np.log10(X1_ARRY), np.log10(X2_ARRY), RHO_ARRY, levels = myLvls)
        cpl = ax1.contour(np.log10(X1_ARRY), np.log10(X2_ARRY), RHO_ARRY,colors='k',levels = myLineLvls)
        ax1.clabel(cpl, fmt='%2.1f', colors='k', fontsize=11)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax1.set_title(r'$\rho$ Contour Plot')
        ax1.set_xlabel(r'$\log_{10}(c_r/s_a)$')
        ax1.set_ylabel(r'$\log_{10}(U_{a,max}/U_r)$')
        plt.show()
        fig.savefig('figures/fig_rho_Uratios_vs_Sratios.pdf')
        
    elif (plotCase == 3):
        myLvls = np.linspace(np.round(np.min(Y_ARRY),1),np.round(np.max(Y_ARRY),1)+0.1,40)
        myLineLvls = np.asarray([0.5,0.80,0.90])
        
        cp = ax1.contourf(np.log10(X1_ARRY), np.log10(X2_ARRY), Y_ARRY, levels = myLvls)
        cpl = ax1.contour(np.log10(X1_ARRY), np.log10(X2_ARRY), Y_ARRY,colors='k',levels = myLineLvls)
        ax1.clabel(cpl, fmt='%2.1f', colors='k', fontsize=11)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax1.set_title(r'Equilibrium $\gamma^*$ Contour Plot')
        ax1.set_xlabel(r'$\log_{10}(c_r/s_a)$')
        ax1.set_ylabel(r'$\log_{10}(U_{a,max}/U_r)$')
        plt.show()
        fig.savefig('figures/fig_gamma_Uratios_vs_Sratios.pdf')
        
    elif (plotCase == 4):
        
        effEvoParam_Arry = effUa_ARRY
        
        cp = ax1.contourf(np.log10(X1_ARRY), np.log10(X2_ARRY), effEvoParam_Arry)
        cpl = ax1.contour(np.log10(X1_ARRY), np.log10(X2_ARRY), effEvoParam_Arry,colors='k')
        ax1.clabel(cpl, fmt='%2.8f', colors='k', fontsize=11)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax1.set_title(r'Effective Parameter Contour Plot')
        ax1.set_xlabel(r'$\log_{10}(c_r/s_a)$')
        ax1.set_ylabel(r'$\log_{10}(U_{a,max}/U_r)$')
        plt.show()
        fig.savefig('figures/fig_effEvoParam_Uratios_vs_Sratios.pdf')
        

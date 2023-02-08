# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description:
Defines the basics functions used in creating grids of the MC chain models.

THIS MODULE NEEDS TO BE COMPLETED
"""
# *****************************************************************************
# libraries
# *****************************************************************************

import numpy as np
import copy as cpy
import pickle 

from evoLibraries.RateOfAdapt import ROA_functions as roaFun

# *****************************************************************************
# Markov Chain Functions
# *****************************************************************************


def get_intersection_rho(vd_i, vc_i, sd_i, Ud_i, Uc_i, sc_i, N_i):
    # This function assumes that the intersection occurs in the 
    # multiple mutations regime. This quantity is irrelevant when in
    # the successional regime since there is no interference between 
    # evolution in traits.
        
    # find index that minimizes |va-vr|, but exclude extinction class (:-1)
    idxMin = np.argmin(np.abs(np.asarray(vd_i[0:-1])-np.asarray(vc_i[0:-1])))
    
    sd = sd_i[idxMin]
    Ud = Ud_i[idxMin]
    sc = sc_i[idxMin]
    Uc = Uc_i[idxMin]
    Npop = N_i[idxMin]

    # calculate the regime IDs for each trait
    #  0: Bad evo parameters
    #  1: successional
    #  2: multiple mutations
    #  3: diffusion
    # -1: regime undetermined, i.e. in transition region   
        
    regimeID_a = roaFun.get_regimeID(Npop,sd,Ud,sd)
    regimeID_r = roaFun.get_regimeID(Npop,sc,Uc,sc)

    
    # calculate the appropriate rho
    if (regimeID_a == 1) or (regimeID_r == 1):
        # either or both in successional regime, no clonal interference
        rho = 0
    
    elif (regimeID_a == 2) and (regimeID_r == 2):
        # both traits in multiple mutations regime
        rho = (sc/np.log(sc/Uc))**2 / (sd/np.log(sd/Ud))**2
        
    elif (regimeID_a == 3) and (regimeID_r == 2):
        # abs trait in diffusion and rel trait in multiple mutations regime
        Da = 0.5*Ud*sd**2
        
        rho = (sc/np.log(sc/Uc))**2 / (Da**(2.0/3.0)/(3*np.log(Da**(1.0/3.0)*Npop)**(2.0/3.0)))               
        
    elif (regimeID_a == 2) and (regimeID_r == 3):
        # rel trait in diffusion and abs trait in multiple mutations regime
        Dr = 0.5*Uc*sc**2
        
        rho = (Dr**(2.0/3.0)/(3*np.log(Dr**(1.0/3.0)*Npop)**(2.0/3.0))) / (sd/np.log(sd/Ud))**2
        
    elif (regimeID_a == 3) and (regimeID_r == 3):
        # both traits in diffusion
        Da = 0.5*Ud*sd**2
        Dr = 0.5*Uc*sc**2
        
        rho = (Dr**(2.0/3.0)/(3*np.log(Dr**(1.0/3.0)*Npop)**(2.0/3.0))) / (Da**(2.0/3.0)/(3*np.log(Da**(1.0/3.0)*Npop)**(2.0/3.0)))
        
    else:
        rho = np.nan
            
    return [rho, sd, Ud, sc, Uc]

#------------------------------------------------------------------------------

def get_intersection_popDensity(vd_i, vc_i, eq_yi):
    # function to calculate the intersection equilibrium density
        
    # find index that minimizes |vd-vc| but exclude extinction class (:-1)
    idxMin = np.argmin(np.abs(np.asarray(vd_i[0:-1])-np.asarray(vc_i[0:-1])))
    
    # Definition of the gamma at intersection in paper
    yiInt = eq_yi[idxMin]
    
    return yiInt

#------------------------------------------------------------------------------
    
def get_contourPlot_arrayData(myOptions):
    # Generic function takes the provided options and generates data needed to
    # creat contour plot of rho and gamma.
    
    # set values of first parameter
    varParam1A = myOptions.varNames[0][0]
    varParam2A = myOptions.varNames[0][1]
    
    varParam1B = myOptions.varNames[1][0]
    varParam2B = myOptions.varNames[1][1]
    
    x1LwrBnd_log10 = np.log10(myOptions.varBounds[0][0]*myOptions.params[varParam1A])
    x1UprBnd_log10 = np.log10(myOptions.varBounds[0][1]*myOptions.params[varParam1A])
    if myOptions.modelType == 'RM':
        x1RefVal_log10 = np.log10(myOptions.params[varParam1B])
    else:
        alpha = myOptions.params[varParam1B]
        de = myOptions.params['b']+1
        d0 = myOptions.params['dOpt']
        sa1_mid = 0.5*(1-alpha)*(de-d0)/((de+(d0-de)*(1-alpha))*(de+(d0-de)*(1-alpha)-1))
        x1RefVal_log10 = np.log10(sa1_mid)
    
    X1_vals = np.logspace(x1LwrBnd_log10, x1UprBnd_log10, num=myOptions.varBounds[0][2])
    X1_ref  = np.logspace(x1RefVal_log10, x1RefVal_log10, num=1                        )
    
    x2LwrBnd_log10 = np.log10(myOptions.varBounds[1][0]*myOptions.params[varParam2A])
    x2UprBnd_log10 = np.log10(myOptions.varBounds[1][1]*myOptions.params[varParam2A])
    x2RefVal_log10 = np.log10(myOptions.params[varParam2B])
    
    X2_vals = np.logspace(x2LwrBnd_log10, x2UprBnd_log10, num=myOptions.varBounds[1][2])
    X2_ref  = np.logspace(x2RefVal_log10, x2RefVal_log10, num=1                        )
    
    X1_ARRY, X2_ARRY = np.meshgrid(X1_vals, X2_vals)
    RHO_ARRY = np.zeros(X1_ARRY.shape)
    Y_ARRY = np.zeros(X1_ARRY.shape)

    # arrays to store effecttive s and U values
    effSa_ARRY = np.zeros(X1_ARRY.shape)
    effUa_ARRY = np.zeros(X1_ARRY.shape)

    effSr_ARRY = np.zeros(X1_ARRY.shape)
    effUr_ARRY = np.zeros(X1_ARRY.shape)
    
    paramsTemp = cpy.copy(myOptions.params)

    # --------------------------------------------------------------------------
    # Calculated rho values for T vs 2nd parameter variable
    # --------------------------------------------------------------------------
    
    # THIS WHOLE SECTION NEEDS TO BE REWRITTEN
    
    for ii in range(int(X1_ARRY.shape[0])):
        for jj in range(int(X2_ARRY.shape[1])):
            
            # set cr and sa values (selection coefficient)
            paramsTemp[varParam1A] = X1_ARRY[ii,jj]
            paramsTemp[varParam1B] = (myOptions.params[varParam1B],X1_ref[0])[myOptions.modelType == 'RM']
            
            # set Ua values and Ur values (mutation coefficient)
            paramsTemp[varParam2A] = X2_ARRY[ii,jj]
            paramsTemp[varParam2B] = X2_ref[0]
            
            # Calculate absolute fitness state space. 
            if myOptions.modelType == 'RM':
                [dMax,di,iMax] = get_absoluteFitnessClasses(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['sd'])
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = get_MChainPopParameters(paramsTemp,di,iMax,myOptions.yi_option)        
            else:
                iStop = np.log(0.01)/np.log(myOptions.params['alpha'])-1  # stop at i steps to get di within 5% of d0, i.e. (di-d0)/(dMax-d0) = 0.05.
                [dMax,di,iMax] = get_absoluteFitnessClassesDRE(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['alpha'],iStop)
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = get_MChainPopParametersDRE(paramsTemp,di,iMax,myOptions.yi_option)
                
            pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
            pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])
            
            # Use s values for pFix until we get sim pFix values can be obtained
            if myOptions.modelType == 'RM':
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                                get_MChainEvoParameters(paramsTemp,di,iMax,pFixAbs_i,pFixRel_i,myOptions.yi_option)
            else:
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                                get_MChainEvoParametersDRE(paramsTemp,di,iMax,pFixAbs_i,pFixRel_i,myOptions.yi_option)                                    
                                
            [RHO_ARRY[ii,jj], effSa_ARRY[ii,jj], effUa_ARRY[ii,jj], effSr_ARRY[ii,jj], effUr_ARRY[ii,jj] ] = \
                                get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i,eq_Ni)   
                            
            Y_ARRY[ii,jj] = get_intersection_popDensity(va_i, vr_i, eq_yi)   
    
    
    with open(myOptions.saveDataName, 'wb') as f:
        pickle.dump([X1_ARRY,X2_ARRY,RHO_ARRY,Y_ARRY,X1_ref,X2_ref, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp,dMax], f)
    
    return None














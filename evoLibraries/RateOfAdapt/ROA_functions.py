# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description:
Defines the basics functions used in all scripts that process matlab
data and create figures in the mutation-driven adaptation manuscript.
"""
# *****************************************************************************
# libraries
# *****************************************************************************

import numpy as np

from evoLibraries import constants as const 

# *****************************************************************************
# FUNCTIONS TO GET QUANTITIES FROM DESAI AND FISHER 2007
# *****************************************************************************

def get_vDF(N,s,U):
    # Calculates the rate of adaptation v, derived in Desai and Fisher 2007
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U)
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    #
    # Output: 
    # v - rate of adaptation
        
    v = s**2*(2*np.log(N*s)-np.log(s/U))/(np.log(s/U)**2)
    
    return v

#------------------------------------------------------------------------------

def get_qDF(N,s,U):
    # Calculates the rate of adaptation v, derived in Desai and Fisher 2007
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U)
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    #
    # Output: 
    # v - rate of adaptation
        
    q = 2*np.log(N*s)/np.log(s/U)
    
    return q

#------------------------------------------------------------------------------

def get_vDF_pFix(N,s,U,pFix):
    # Calculates the rate of adaptation v, using a heuristic version of Desai 
    # and Fisher 2007 argument, but without the asumption that p_fix ~ s, i.e. 
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U) and finally p_fix
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix -  probability of fixation
    #
    # Output: 
    # v - rate of adaptation
        
    v = s**2*(2*np.log(N*pFix)-np.log(s/U))/(np.log(s/U)**2)
    
    return v


#------------------------------------------------------------------------------

def get_vSucc_pFix(N,s,U,pFix):
    # Calculates the rate of adaptation v for the successional regime
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix - probability of fixation
    #
    # Output: 
    # v - rate of adaptation
        
    v = N*U*pFix*s
    
    return v

#------------------------------------------------------------------------------

def get_vOH(N,s,U):
    # Calculates the rate of adaptation v, using a heuristic version of Desai 
    # and Fisher 2007 argument, but without the asumption that p_fix ~ s, i.e. 
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U) and finally p_fix
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    #
    # Output: 
    # v - rate of adaptation

    D = 0.5*U*s**2     
    
    v = D**0.6667*(np.log(N*D**0.3333))**0.3333
    
    return v

#------------------------------------------------------------------------------

def get_rateOfAdapt(N,s,U,pFix):
    # Calculates the rate of adaptation v, but checks which regime applies.
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix -  probability of fixation
    #
    # Output: 
    # v - rate of adaptation
    #
    
    # check that selection coefficient, pop size and beneficial mutation rate 
    # are valid parameters
    
    regimeID = get_regimeID(N,s,U,pFix)        
    
    if ( (regimeID == 0) or (regimeID == -1) ):
        # bad evolutionary parameters with either N,s,U,pFix = 0 (regID == 0)
        # or regime could not be determined (regID == -1)
        v = 0
    elif regimeID == 1:
        # successional regime
        v = get_vSucc_pFix(N,s,U,pFix)        
        
    elif regimeID == 2:
        # multiple mutation regime
        v = get_vDF_pFix(N,s,U,pFix)
        
    elif regimeID == 2: 
        # diffusive mutations regime
        v = get_vOH(N,s,U)        
        
    return v

#------------------------------------------------------------------------------
    
def get_regimeID(N,s,U,pFix):
    # Calculates the rate of adaptation v, but checks which regime applies.
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix -  probability of fixation
    #
    # Output: 
    # v - rate of adaptation
    #
    
    # check that selection coefficient, pop size and beneficial mutation rate 
    # are valid parameters
    if (s <= 0) or (N <= 0) or (U <= 0) or (pFix <= 0):
        # bad evolutionary parameters
        regID = 0
        return regID
    
    # Calculate mean time between establishments
    Test = 1/N*U*pFix
    
    # Calculate mean time of sweep
    Tswp = (1/s)*np.log(N*pFix)
        
    # calculate rate of adaptation based on regime
    if (Test * const.CI_TIMESCALE_TRANSITION >= Tswp):
        # successional, establishment time scale exceeds sweep time scale
        regID = 1
        
    elif (s > const.MM_REGIME_MULTIPLE*U) and (Test < const.CI_TIMESCALE_TRANSITION*Tswp):
        # multiple mutations, selection time scale smaller than  mutation time scale
        regID = 2
        
    elif (s <= const.MM_REGIME_MULTIPLE*U) and (Test < const.CI_TIMESCALE_TRANSITION*Tswp):
        # diffusive mutations, 
        regID = 3
    
    else:
        # regime undetermined
        regID = -1
    
    return regID

#------------------------------------------------------------------------------
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

import matplotlib.tri as tri

# *****************************************************************************
# Markov Chain Functions
# *****************************************************************************
    
def get_contourPlot_arrayData(X,Y,Z,nGridCt):
    # Generic function that takes the provided MC grid data and forms the
    # arrays that can be used to build a contour plot.
    
    # Create grid values first.
    xi = np.linspace(X.min(), X.max(), nGridCt)
    yi = np.linspace(Y.min(), Y.max(), nGridCt)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(X.flatten(), Y.flatten())
    interpolator = tri.LinearTriInterpolator(triang, Z.flatten())
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interpolator(Xi, Yi)
    
    return [Xi,Yi,Zi]

#------------------------------------------------------------------------------

def calculate_evoRates_rho(N, evoParams_Trait_1, evoParams_Trait_2):
    # This function calculate the rho parameter defined in the manuscript,
    # which measures the relative changes in evolution rates due to increases
    # in max available territory parameter. 
    #
    # NOTE: This provides a generic function for calculating rho defined in the
    #       manuscript. We don't include it in the rate of adaptation library 
    #       because this quantity only makes sense to calculate in the context
    #       of an MC model.
    
    # Input evoParams_Trait_# is dictionary with following terms
    # 's'     - selection coefficient
    # 'pFix'  - probability of fixation 
    # 'U'     - beneficial mutation rate
    # 'regID' - regime ID for rate of adaptation
    #           List of possible regime IDs 
    #            0: Bad evo parameters (either U, s, N, pFix == 0)
    #            1: successional
    #            2: multiple mutations
    #            3: diffusion
    #           -1: regime undetermined, i.e. in transition region   
        
    # load evo parameters for trait 1 into variables to make calculations 
    # easier to read
    s_1         = evoParams_Trait_1['s']
    pFix_1      = evoParams_Trait_1['pFix']
    U_1         = evoParams_Trait_1['U']
    evoRegID_1  = evoParams_Trait_1['regID']
    
    # load evo parameters for trait 2 into variables to make calculations 
    # easier to read
    s_2         = evoParams_Trait_2['s']
    pFix_2      = evoParams_Trait_2['pFix']
    U_2         = evoParams_Trait_2['U']
    evoRegID_2  = evoParams_Trait_2['regID']    
    
    # calculate the appropriate rho
    if (evoRegID_1 == 1) or (evoRegID_2 == 1):
        # either or both in successional regime, no clonal interference
        rho = 0
    
    elif (evoRegID_1 == 2) and (evoRegID_2 == 2):
        # both traits in multiple mutations regime
        rho = (s_2/np.log(s_2/U_2))**2 / (s_1/np.log(s_1/U_1))**2
        
    elif (evoRegID_1 == 3) and (evoRegID_2 == 2):
        # abs trait in diffusion and rel trait in multiple mutations regime
        D_1 = 0.5*U_1*s_1**2
        
        rho = (s_2/np.log(s_2/U_2))**2 / (D_1**(2.0/3.0)/(3*np.log(D_1**(1.0/3.0)*N)**(2.0/3.0)))               
        
    elif (evoRegID_1 == 2) and (evoRegID_2 == 3):
        # rel trait in diffusion and abs trait in multiple mutations regime
        D_2 = 0.5*U_2*s_2**2
        
        rho = (D_2**(2.0/3.0)/(3*np.log(D_2**(1.0/3.0)*N)**(2.0/3.0))) / (s_1/np.log(s_1/U_1))**2
        
    elif (evoRegID_1 == 3) and (evoRegID_2 == 3):
        # both traits in diffusion regime
        D_1 = 0.5*U_1*s_1**2
        D_2 = 0.5*U_2*s_2**2
        
        rho = (D_2**(2.0/3.0)/(3*np.log(D_2**(1.0/3.0)*N)**(2.0/3.0))) / (D_1**(2.0/3.0)/(3*np.log(D_1**(1.0/3.0)*N)**(2.0/3.0)))
        
    else:
        rho = np.nan
            
    return rho

#------------------------------------------------------------------------------

def calculate_v_intersections(vDiff):
    
    return [crossings, cross_types]

-----------------------------------------------------------
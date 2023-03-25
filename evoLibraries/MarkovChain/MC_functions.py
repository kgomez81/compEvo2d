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
    
    if (s_1 * pFix_1 * U_1 * s_2 * pFix_2 * U_2 == 0):
        return np.nan
    
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
    # calculate_v_intersections() determines points where the vDiff array cross
    # the zero axis, and also provide what type of crossing occurs.
    
    # there are a couple of cases to consider
    #   Group I - Pure crossing with crossing state occuring such that v1 = v2 exactly 
    #
    #   Group II - Pcrossings at endpoints of the chain
    #   2A. crossings in the interior states 
    crossings   = []
    cross_types = []
    
    vDiffSgn = np.sign(vDiff)
    
    cross_1 = np.where(vDiffSgn                      == 0)[0]
    cross_2 = np.where(vDiffSgn[0:-1] + vDiffSgn[1:] == 0)[0]
    
    # check cross type 1 where v1 == v2
    for ii in range(len(cross_1)):
        idx = cross_1[ii]
        
        if idx == 0:
            crossings   = crossings   + [ idx             ]
            cross_types = cross_types + [2*vDiffSgn[idx+1]]
            
        elif idx == len(vDiffSgn)-1:
            crossings   = crossings   + [ idx              ]
            cross_types = cross_types + [-2*vDiffSgn[idx-1]]
            
        else:
            if (vDiffSgn[idx-1] != vDiffSgn[idx+1]):
                crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])
                
                crossings   = crossings   + [idx       ]
                cross_types = cross_types + [crossSign ]
    
    # check cross type 1 where v1 == v2
    for ii in range(len(cross_2)):
    
        idx = cross_2[ii]
        
        if (idx == 0):
            minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
            
            crossings   = crossings   + [ idx + minIdx     ]
            cross_types = cross_types + [ 2*vDiffSgn[idx+1]]
            
        elif (idx == len(vDiffSgn)-1):
            minIdx = np.argmin([vDiff[idx],vDiff[idx-1]])
            
            crossings   = crossings   + [ idx - minIdx     ]
            cross_types = cross_types + [-2*vDiffSgn[idx-1]]
            
        else:
            if (vDiffSgn[idx] != vDiffSgn[idx+1]):
                crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])
                minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
                
                crossings   = crossings   + [idx + minIdx ]
                cross_types = cross_types + [crossSign    ]
                
    crossings   = np.asarray(crossings)
    cross_types = np.asarray(cross_types)
        
    return [crossings, cross_types]

#------------------------------------------------------------------------------
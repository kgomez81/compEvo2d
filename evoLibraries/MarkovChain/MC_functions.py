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

def calculate_evoRates_rho(N_eq, evoParams_Trait_1, evoParams_Trait_2):
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
    
    DebugFlag  = False
    
    # When using this function with absolute and relative fitness trains, note
    # that definition of "Rho" from manuscript is 
    #
    #                 Rho = del_vc / del_vd
    #
    # i.e. first parameters must be for absolute trait, and second parameters 
    # must be for relative trait
    #
    # - Case "Rho > 1": change in vc is larger, causing shift back in abs fitness
    # - Case "Rho < 1": change in vc is smaller causing shift forward in abs fitness
            
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
    
    if (s_1 * pFix_1 * U_1 * s_2 * pFix_2 * U_2 * N_eq == 0):
        return np.nan
    
    # calculate the appropriate rho
    if (evoRegID_1 == 1) or (evoRegID_2 == 1):
        # either or both in successional regime, no clonal interference
        rho = 0
        
    elif (evoRegID_1 == 2) and (evoRegID_2 == 2):
        # both traits in multiple mutations regime
        rho = (s_2/np.log(s_2/U_2))**2 / (s_1/np.log(s_1/U_1))**2
        
        if (rho > 1) and DebugFlag:
            print("   RHO = %f" % (rho))
            print("   del_vc = %f" % (s_2/np.log(s_2/U_2))**2)
            print("   sc = %f" % (s_2))
            print("   Uc = %f" % (U_2))
            print("   N  = %f" % (N_eq))      
            
            print("   del_vd = %f" % (s_1/np.log(s_1/U_1))**2)
            print("   sd = %f" % (s_1))
            print("   Ud = %f" % (U_1))
            print("   N  = %f" % (N_eq))      
            
    elif (evoRegID_1 == 3) and (evoRegID_2 == 2):
        # abs trait in diffusion and rel trait in multiple mutations regime
        D_1 = 0.5*U_1*s_1**2
        
        rho = (s_2/np.log(s_2/U_2))**2 / (D_1**(2.0/3.0)/(3*np.log(D_1**(1.0/3.0)*N_eq)**(2.0/3.0)))               
        
    elif (evoRegID_1 == 2) and (evoRegID_2 == 3):
        # rel trait in diffusion and abs trait in multiple mutations regime
        D_2 = 0.5*U_2*s_2**2
        
        rho = (D_2**(2.0/3.0)/(3*np.log(D_2**(1.0/3.0)*N_eq)**(2.0/3.0))) / (s_1/np.log(s_1/U_1))**2
        
    elif (evoRegID_1 == 3) and (evoRegID_2 == 3):
        # both traits in diffusion regime
        D_1 = 0.5*U_1*s_1**2
        D_2 = 0.5*U_2*s_2**2
        
        rho = (D_2**(2.0/3.0)/(3*np.log(D_2**(1.0/3.0)*N_eq)**(2.0/3.0))) / (D_1**(2.0/3.0)/(3*np.log(D_1**(1.0/3.0)*N_eq)**(2.0/3.0)))
        
    else:
        rho = np.nan
    
    if DebugFlag:
        print("(rho=%f, regID_d=%f, regID_c=%f)" % (rho,evoRegID_1,evoRegID_1))        
        
    return rho

#------------------------------------------------------------------------------

def calculate_v_intersections(vDiff):
    # calculate_v_intersections() determines points where the vDiff array cross
    # the zero axis, and also provide what type of crossing occurs.
    
    # To find the intersections, we need to consider a few possible cases
    #
    #   Case 01 - v's cross through a state, such that v1 = v2 exactly at some 
    #             state, and "v1-v2" differ in sign in the adjacent states to 
    #             where v1 = v2
    #
    #   Case 02 - v's cross between a state, such that "v1-v2" differ in sign
    #             from one state to the next
    #
    #   Case 03 - intersection of v's occurs at endpoints of the state space.
    crossings   = []
    cross_types = []
    
    # NOTE:     vDiff = va-vr or = va-ve 
    vDiffSgn = np.sign(vDiff)       
    
    # select indices where sign is 0, i.e. va = v
    cross_1 = np.where(vDiffSgn                      == 0)[0]   
    
    # select indices where sign of va - v changes
    cross_2 = np.where(vDiffSgn[0:-1] + vDiffSgn[1:] == 0)[0]
    
    # check all crossings of type 1 where va == v
    for ii in range(len(cross_1)):
        idx = cross_1[ii]
        
        if idx == 0:
            # crossing at start of state space
            crossings   = crossings   + [ idx             ]     # add index of cross to list
            cross_types = cross_types + [2*vDiffSgn[idx+1]]     # save cross type
                                                                #   =  2 if "iExt: va == v" => "iExt+1: va > v", i.e. va cross up v
                                                                #   = -2 if "iExt: va == v" => "iExt+1: va < v", i.e. va cross down v
        elif idx == len(vDiffSgn)-1:
            # crossing at end of state space
            crossings   = crossings   + [ idx              ]    # add index of cross to list
            cross_types = cross_types + [-2*vDiffSgn[idx-1]]    # save cross type
                                                                #   = -2 if "iMax-1: va > v" => "iMax: va == v", i.e. va cross down v
                                                                #   =  2 if "iMax-1: va < v" => "iMax: va == v", i.e. va cross up v
        else:
            # crossing between min and max indices of state space
            # NOTE: this type of crossing is important to the analysis
            if (vDiffSgn[idx-1] != vDiffSgn[idx+1]):
                crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])    # calc cross sign
                                                                            #   =  1 if va > v before cross & va < v after cross
                                                                            #   = -1 if va < v before cross & va > v after cross
                
                crossings   = crossings   + [idx       ]        # add index of cross to list
                cross_types = cross_types + [crossSign ]        # save cross type
    
    # check all crossings of type 2 where sign(va(i)-v2(i)) != sign(va(i+1)-v2(i+1))
    # this will not include crosses of type 1.
    for ii in range(len(cross_2)):
    
        idx = cross_2[ii]
        
        if (idx == 0):
            # crossing near the start of the state space
            # select as the crossing, idx which minimizes the v-difference
            minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
            
            crossings   = crossings   + [ idx + minIdx     ]    # add index of cross to list
            cross_types = cross_types + [ 2*vDiffSgn[idx+1]]    # save cross type
                                                                #   =  2 if "iExt: va == v" => "iExt+1: va > v", i.e. va cross up v
                                                                #   = -2 if "iExt: va == v" => "iExt+1: va < v", i.e. va cross down v
            
        elif (idx == len(vDiffSgn)-1):
            # crossing near the end of the state space
            # select as the crossing, idx which minimizes the v-difference
            minIdx = np.argmin([vDiff[idx],vDiff[idx-1]])
            
            crossings   = crossings   + [ idx - minIdx     ]     # add index of cross to list
            cross_types = cross_types + [-2*vDiffSgn[idx-1]]     # save cross type
                                                                 #   = -2 if "iMax-1: va > v" => "iMax: va == v", i.e. va cross down v
                                                                 #   =  2 if "iMax-1: va < v" => "iMax: va == v", i.e. va cross up v
            
        else:
            # crossing between min and max indices of state space
            # NOTE: this type of crossing is important to the analysis
            if (vDiffSgn[idx] != vDiffSgn[idx+1]):
                crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])    # calc cross sign
                minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])               # select as cross, idx that minizes the v-difference
                
                crossings   = crossings   + [idx + minIdx ]     # add index of cross to list
                cross_types = cross_types + [crossSign    ]     # save cross type
        
    # summary of outputs:
    # crossings     = list of indices for all crossings
    #
    # cross_types   = +/- 2 cross at endpoints of state space, 
    #               = +/- 1 cross between state space endpoints
    #                   + implies va crosses up above v
    #                   - implies va crosses down below v
    crossings   = np.asarray(crossings)
    cross_types = np.asarray(cross_types)
        
    return [crossings, cross_types]

#------------------------------------------------------------------------------
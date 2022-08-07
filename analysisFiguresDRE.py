# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:28:52 2022

@author: dirge
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import evo_library as myfun            # my functions in a seperate file
from mpl_toolkits.mplot3d import Axes3D
import copy as cpy
import pickle 
import matplotlib.tri as tri
from mpl_toolkits import mplot3d

import evo_library as myfun            # evo functions in seperate file

# --------------------------------------------------------------------------
#                               Load data
# --------------------------------------------------------------------------

# load the array data for rho and variable parameter
with open('outputs/dat_rho_eff_UvsS_evoExp_DRE_02.pickle', 'rb') as f:
    [P1_ARRY, P2_ARRY, RHO_ARRY, Y_ARRY, P1_valMid, P2_valMid, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp, dMax] = pickle.load(f)


# --------------------------------------------------------------------------
#                               check entry
# --------------------------------------------------------------------------
ii = 40
jj = 2
importlib.reload(myfun)

plotterVaVr_DRE_fromArray(myOptions,ii,jj,P1_ARRY,P2_ARRY,P1_valMid,P2_valMid,0)

def plotterVaVr_DRE_fromArray(myOptions,ii,jj,X1_ARRY,X2_ARRY,X1_ref,X2_ref,myPlotCase):
    
    paramsTemp = cpy.copy(myOptions.params)
    
    # set values of first parameter
    varParam1A = myOptions.varNames[0][0]
    varParam2A = myOptions.varNames[0][1]
    
    varParam1B = myOptions.varNames[1][0]
    varParam2B = myOptions.varNames[1][1]
    
    # set cr and sa values
    paramsTemp[varParam1A] = X1_ARRY[ii,jj]
    paramsTemp[varParam1B] = (myOptions.params[varParam1B],X1_ref[0])[myOptions.modelType == 'RM']
    
    # set Ua values and Ur values
    paramsTemp[varParam2A] = X2_ARRY[ii,jj]
    paramsTemp[varParam2B] = X2_ref[0]
    
    # Calculate absolute fitness state space. 
    if myOptions.modelType == 'RM':
        [dMax,di,iMax] = myfun.get_absoluteFitnessClasses(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['sa'])
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(paramsTemp,di,iMax,myOptions.yi_option)        
    else:
        iStop = np.log(0.000001)/np.log(myOptions.params['alpha'])-1  # stop at i steps to get di within 5% of d0, i.e. (di-d0)/(dMax-d0) = 0.05.
        [dMax,di,iMax] = myfun.get_absoluteFitnessClassesDRE(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['alpha'],iStop)
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParametersDRE(paramsTemp,di,iMax,myOptions.yi_option)
        
    pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
    pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])
    
    # Use s values for pFix until we get sim pFix values can be obtained
    if myOptions.modelType == 'RM':
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                        myfun.get_MChainEvoParameters(paramsTemp,di,iMax,pFixAbs_i,pFixRel_i,myOptions.yi_option)
    else:
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                        myfun.get_MChainEvoParametersDRE(paramsTemp,di,iMax,pFixAbs_i,pFixRel_i,myOptions.yi_option)                                    
                        
    fig1, ax1 = plt.subplots(1,1,figsize=[7,6])
    if myPlotCase == 0:
        ax1.scatter(state_i,va_i,color="blue",s=8,label=r'$v_a$')
        ax1.scatter(state_i,vr_i,color="red",s=8,label=r'$v_r$')
        ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
        ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
        ax1.legend(fontsize = 14,ncol=1,loc='lower right')
    elif myPlotCase == 1:
        ax1.scatter(state_i,sa_i,color="blue",s=8,label=r'$sa$')
        ax1.scatter(state_i,sr_i,color="red",s=8,label=r'$sr$')
        ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
        ax1.set_ylabel(r'selection coefficients',fontsize=20,labelpad=8)
        ax1.legend(fontsize = 14,ncol=1,loc='lower right')
    elif myPlotCase == 2:
        ax1.scatter(state_i,Ua_i,color="blue",s=8,label=r'$Ua$')
        ax1.scatter(state_i,Ur_i,color="red",s=8,label=r'$Ur$')
        ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
        ax1.set_ylabel(r'Mutation rates',fontsize=20,labelpad=8)
        ax1.legend(fontsize = 14,ncol=1,loc='lower right')
    elif myPlotCase == 3:
        ax1.scatter(state_i,np.log10(eq_Ni),color="blue",s=8,label=r'$N^*$')
        ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
        ax1.set_ylabel(r'Equilibrium Pop Size',fontsize=20,labelpad=8)
        ax1.legend(fontsize = 14,ncol=1,loc='lower right')
    elif myPlotCase == 4:
        print(len(state_i))
        print(len(di[:-1]))
        ax1.scatter(state_i,di[:-1],color="blue",s=8,label=r'$N^*$')
        ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
        ax1.set_ylabel(r'death term value',fontsize=20,labelpad=8)
        ax1.legend(fontsize = 14,ncol=1,loc='lower right')    
    plt.show()
    
    return 0


# square array is built with first two parmeters, and second set are held constant.
# varNames[0][0] stored as X1_ARRY
# varNames[1][0] stored as X1_ref
# varNames[0][1] stored as X2_ARRY
# varNames[1][1] stored as X2_ref
varNames        = [['cr','Ua'],['alpha','Ur']]  

# varBounds values define the min and max bounds of parameters that are used to 
# define the square grid. 
# varBounds[j][0] = min Multiple of parameter value in file (Xj variable)
# varBounds[j][1] = max Multiple of parameter value in file (Xj variable)
# varBounds[j][2] = number of increments from min to max (log scale) 
gridCnt         = 41
varBounds       = [[1e-1,1e+1,gridCnt],[1e-1,1e+1,gridCnt]]
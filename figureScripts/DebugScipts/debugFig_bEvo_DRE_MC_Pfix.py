# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:59:26 2023

@author: kgomez

This script was generate to test the pfix estimations

"""
# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time

import evoLibraries.LotteryModel.LM_pFix_FSA as lmPfix
import evoLibraries.LotteryModel.LM_pFix_MA as lmPfix2

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d')

from evoLibraries.MarkovChain import MC_array_class as mcArry
# from evoLibraries.MarkovChain import MC_functions as mcFun

def getScatterData(X,Y,Z):
    
    x = []
    y = []
    z = []
    
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            
            # removed bad data
            xGood = not np.isnan(X[ii,jj]) and not np.isinf(X[ii,jj])
            yGood = not np.isnan(Y[ii,jj]) and not np.isinf(Y[ii,jj])
            zGood = not np.isnan(Z[ii,jj]) and not np.isinf(Z[ii,jj])
            
            if xGood and yGood and zGood:
                x = x + [ X[ii,jj] ]
                y = y + [ Y[ii,jj] ]
                z = z + [ Z[ii,jj] ]
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    return [x,y,z]

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.getcwd()+'/inputs/evoExp_DRE_bEvo_06_parameters.csv'
modelType = 'DRE'
absFitType = 'bEvo'

# set list of variable names that will be used to specify the grid
# and the bounds with increments needed to define the grid.
# varNames[0] = string with dictionary name of evo model parameter
# varNames[1] = string with dictionary name of evo model parameter
varNames       = ['Ua','cp']

# varBounds values define the min and max bounds of parameters that are used to 
# define the square grid. First index j=0,1 (one for each evo parameter). 
# varBounds[0]    = list of base 10 exponentials to use in forming the parameter 
#                   grid for X1
# varBounds[1]    = list of base 10 exponentials to use in forming the parameter 
#                   grid for X2
# NOTE: both list should include 0 to represent the center points of the grid.
#       For example, [-2,-1,0,1,2] would designate [1E-2,1E-1,1E0,1E1,1e2].
#       Also note that the entries don't have to be integers.
nArry     = 11

Ua_Bnds = np.linspace(-3, 3, nArry)
cp_Bnds = np.linspace(-1, 1, nArry)   # cannot exceed ~O(10^-1) for pFix estimates

varBounds = [Ua_Bnds, cp_Bnds]

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# generate grid
tic = time.time()
mcModels = mcArry.mcEvoGrid(paramFilePath, modelType, absFitType, varNames, varBounds)
print(time.time()-tic)

#%% ------------------------------------------------------------------------
# construct plot variables
# --------------------------------------------------------------------------

X = np.log10(mcModels.eff_sc_ij / mcModels.eff_sa_ij)   # sc/sd
Y = np.log10(mcModels.eff_Ua_ij / mcModels.eff_Uc_ij)   # Ud/Uc
Z = mcModels.rho_ij                                     # rho

[x,y,z] = getScatterData(X,Y,Z)

zRange = np.max(np.abs(z-1))


#%%-------------------------------------------------------------------------


ii=7
jj=3

mcTestEvoOptions = mcModels.get_params_ij(ii,jj)
mcTestModel = mcDRE.mcEvoModel_DRE(mcTestEvoOptions)

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[5,7])
ax1.scatter(mcTestModel.bi,mcTestModel.sa_i)
ax1.scatter(mcTestModel.bi,mcTestModel.pFix_a_i)
ax2.scatter(mcTestModel.bi,mcTestModel.eq_yi)
ax3.scatter(mcTestModel.bi,mcTestModel.eq_yi*max(mcTestModel.di))

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[5,7])
ax1.scatter(mcTestModel.state_i,mcTestModel.sa_i)
ax1.scatter(mcTestModel.state_i,mcTestModel.pFix_a_i)
ax2.scatter(mcTestModel.state_i,mcTestModel.eq_yi)
ax3.scatter(mcTestModel.state_i,mcTestModel.eq_yi*max(mcTestModel.di))

#%% -------------------------------------------------------------------------


def popDens(bi,d):
    
    yi = np.zeros(bi.shape)
    
    for ii in range(bi.shape[0]):
        yi[ii] = (1-np.exp(-(bi[ii]-bi[0])/d))/(d-np.exp(-(bi[ii]-bi[0])/d))
    return yi

ii = 200
b = [mcTestModel.bi[ii], mcTestModel.bi[ii+1]]
d = [mcTestModel.di[ii], mcTestModel.di[ii+1]]
c = np.array( [1, 1] )
#c = np.array( [1, 1+mcTestModel.params['cp'] ] )  # mutation in c-trait

kMax = 15

t = time.time()
pFixFSA = lmPfix.calc_pFix_FSA(mcTestModel.params,b,d,c,10)
elapsedFSA = time.time() - t

t = time.time()
pFixMA = lmPfix2.calc_pFix_MA(mcTestModel.params,b,d,c,1000,1000,0)
elapsedMA = time.time() - t

(pFixFSA,elapsedFSA,pFixMA,elapsedMA,mcTestModel.sa_i[ii])

mcTestModel.pFix_a_i[ii]

dEvo = max(mcTestModel.di)

fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=[12,12])
ax1.plot(mcTestModel.state_i,mcTestModel.pFix_a_i,c='b')
ax1.plot(mcTestModel.state_i,mcTestModel.sa_i,c='r')
ax2.plot(mcTestModel.state_i,np.log10(1-dEvo*mcTestModel.eq_yi),c='g')
ax2.plot(mcTestModel.state_i,np.log10(1-dEvo*popDens(mcTestModel.bi,dEvo)),c='m')
ax3.scatter(mcTestModel.state_i,mcTestModel.bi,c='k')
ax4.plot(mcTestModel.bi,mcTestModel.eq_yi,c='g')
ax4.plot(mcTestModel.bi,popDens(mcTestModel.bi,dEvo),c='m')
ax5.scatter(mcTestModel.state_i,mcTestModel.eq_yi,c='g')
ax5.scatter(mcTestModel.state_i,popDens(mcTestModel.bi,dEvo),c='m')
ax6.plot(mcTestModel.state_i,mcTestModel.pFix_c_i,c='b')
ax6.plot(mcTestModel.state_i,mcTestModel.sc_i,c='r')

#%%

ii = 300
b = [mcTestModel.bi[ii], mcTestModel.bi[ii]]
d = [mcTestModel.di[ii], mcTestModel.di[ii]]
c = np.array( [1, 1+mcTestModel.params['cp']] )
#c = np.array( [1, 1+mcTestModel.params['cp'] ] )  # mutation in c-trait

kMax = 15

t = time.time()
pFixFSA = lmPfix.calc_pFix_FSA(mcTestModel.params,b,d,c,10)
elapsedFSA = time.time() - t

t = time.time()
pFixMA = lmPfix2.calc_pFix_MA(mcTestModel.params,b,d,c,1000,1000,0)
elapsedMA = time.time() - t

(pFixFSA,elapsedFSA,pFixMA,elapsedMA,mcTestModel.sc_i[ii])

mcTestModel.pFix_a_i[ii]

dEvo = max(mcTestModel.di)

fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=[12,12])
ax1.plot(mcTestModel.state_i,mcTestModel.pFix_a_i,c='b')
ax1.plot(mcTestModel.state_i,mcTestModel.sa_i,c='r')
ax2.plot(mcTestModel.state_i,np.log10(1-dEvo*mcTestModel.eq_yi),c='g')
ax2.plot(mcTestModel.state_i,np.log10(1-dEvo*popDens(mcTestModel.bi,dEvo)),c='m')
ax3.scatter(mcTestModel.state_i,mcTestModel.bi,c='k')
ax4.plot(mcTestModel.bi,mcTestModel.eq_yi,c='g')
ax4.plot(mcTestModel.bi,popDens(mcTestModel.bi,dEvo),c='m')
ax5.scatter(mcTestModel.state_i,mcTestModel.eq_yi,c='g')
ax5.scatter(mcTestModel.state_i,popDens(mcTestModel.bi,dEvo),c='m')
ax6.plot(mcTestModel.state_i,mcTestModel.pFix_c_i,c='b')
ax6.plot(mcTestModel.state_i,mcTestModel.sc_i,c='r')
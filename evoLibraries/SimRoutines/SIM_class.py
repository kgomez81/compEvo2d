# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:09:10 2022

@author: Kevin Gomez

This file defines the simualtion class, which is used to run full simulations of 
evoluation with selection defined by the variable density lottery model

"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries import evo_library as myfun            # my functions in a seperate file
from evoLibraries import sim_library as mysim            # my functions in a seperate file

from numpy.polynomial import Polynomial


# set up parameter values
# The parameter file is read and a dictionary with their values is generated.
workDir = 'D:/Documents/GitHub/compEvo2d/'
paramFile = workDir + 'inputs/evoExp_RM_01_parameters.csv'
params = myfun.read_parameterFile(paramFile)

[dMax,di,iExt] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

yi_option = 3  

[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(params,di,iExt,yi_option)

iCheck      = 171;
b           = params['b']        
d           = np.array([2.388760  ,  2.26938698848899])
c           = np.array([1         ,         1 ])
T           = params['T']

wldTypeInitPop = int(eq_Ni[iCheck]-1)

init_pop    = np.array([wldTypeInitPop,         1 ])

# ------------------------------------------------------------------------------

# testing pFix function
kMax=10
pfix = myfun.get_pFix_dTrait_FSA(b,T,d,kMax)



poly = Polynomial(cffPfix)
x = np.asarray([1-0.1*ii/100 for ii in range(101)])
y = poly(x)

plt.plot(x,y)

# -------------------------------------------------------------
yEq = myfun.get_eqPopDensity(b,d[0],3)

# mutant lineage's rate of acquisition for territories (one mutant adult)
lMut_1 = ( (1-yEq)/yEq ) * ( 1-np.exp(-b*yEq) ) * np.exp(-b/T)

[ myfun.get_pFixCoeffFSA_1Mut(ii,di[1],lMut_1) for ii in range(kMax+1) ]
# ------------------------------------------------------------------------------

U           = T-np.sum(init_pop)
m           = b*init_pop*U/T
l           = m/U

# testing the Ai solution in the evolibrary
Ai_ans = mysim.calculate_Ai_term(m,c,U)

# testing the Ri solution in the evolibrary
Ri_ans = mysim.calculate_Ri_term(m,c,U)

# testing change in population
init_pop + mysim.deltnplus(m,c,U)

pop = (init_pop + mysim.deltnplus(m,c,U))/d

Ai_ans
Ri_ans
pop
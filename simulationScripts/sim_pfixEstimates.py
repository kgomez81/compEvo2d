# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:08:31 2022
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries import evo_library as myfun            # my functions in a seperate file
from evoLibraries import sim_library as mysim            # my functions in a seperate file

# set up parameter values
# The parameter file is read and a dictionary with their values is generated.
workDir = 'D:/Documents/GitHub/compEvo2d/'
paramFile = workDir + 'inputs/evoExp_RM_01_parameters.csv'
params = myfun.read_parameterFile(paramFile)


# Calculate absolute fitness state space. This requires specificying:
# dMax  - max size of death term that permits non-negative growth in abundances
# di    - complete of death terms of the various absolute fitness states
# iExt  - extinction class
[dMax,di,iExt] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

# set root solving option for equilibrium densities
# (1) low-density analytic approximation 
# (2) high-density analytic approximation
# (3) root solving numerical approximation
yi_option = 3  

# get pop parameters
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(params,di,iExt,yi_option)

# set sim parameters                                        
iCheck      = 171;
# d           = np.array([di[iCheck+1]  , di[iCheck-15]])
# d           = np.array([2.388760  ,  2.3270015947896763])
d           = np.array([2.388760  ,  2.2270015947896763])
c           = np.array([1         ,         1 ])
nPfix       = 10000
fixThrshld  = 1e3

# First apply deaths to wild type, i.e. We assume that the mutant survived competition to 
# enter the reproductive state. However, the wild type population still must go throught he
# absolute death phase.
wldTypeInitPop = eq_Ni[iCheck+1]-1;
wldTypeInitPop = int(wldTypeInitPop)
init_pop    = np.array([wldTypeInitPop,         1 ])

# run simulation
# print("pop array init = (%i,%i)" % tuple(init_pop))
pfixEst = mysim.simulation_popEvo_pFixEst(params,init_pop,d,c,nPfix,fixThrshld)

print("%f, %f" % (d[1],pfixEst))

# ---- analysis --------------
def getCDF(x):
    N = len(x)
    y = [(i+1)/(N+1) for i in range(N)]
    return [np.sort(x[:,0]),y]

        

    
    
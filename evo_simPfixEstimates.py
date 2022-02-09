# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 2022
@author: kgomez81
"""

import time
import numpy as np
import evo_library as myfun  

# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp01_parameters_VaVeIntersect.csv'
params = myfun.read_parameterFile(paramFile)

# Calculate absolute fitness state space. This requires specificying:
[dMax,di,iExt] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

# option for determining equilibrium density
yi_option = 3   

# get Markov chain population parameters
[state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(params,di,iExt,yi_option)

# set the number of fix estimates that must be calculated and define arrays to 
# store the values.
# pFixAbs_i estimates range between k=1,2,...,iExt, i.e. mutants from k->k-1
# pFixRel_i estimates range between k=1,2,...,iExt, i.e. mutants in comp trait
#           at state k to compete with absolute fitness mutations.
nEst = len(di)-1
pFixAbs_i = np.zeros([nEst,1])
pFixRel_i = np.zeros([nEst,1])

samp = 10
sub_sample = len(di)/part - 1
t = time.time()

# simulate evolution of a population to estimate pfix
# pFix for mutations from the extinction class is not calculated. That should
# be one since the extinction class is not viable
pFixAbs_i[0,0] = np.zeros([nEst,1])

for ii in range(1,len(nEst)):

    ###### Relative Fitness pFix #######
    # set initial pop sizes
    pop = 
    
    # calcualte the d and c's
    d = di[ii]
    c = np.array([1,1])
    
    # estimate pFix values
    
    ###### Relative Fitness pFix #######
    # set initial pop sizes
    
    # calcualte the d and c's

    # estimate pFix values
    
# write estimate pFix values 
fwrite_abs = open("evoExp01_absPfix.csv","a")
fwrite_rel = open("evoExp01_absPfix.csv","a") 



print(time.time() - t)  


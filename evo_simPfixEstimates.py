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

# Calculate absolute fitness state space. 
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
pFixAbs_i = np.zeros([nEst])
pFixRel_i = np.zeros([nEst])

nPfix = 100
fixThrshld = 1000

# simulate evolution of a population to estimate pfix
# pFix for mutations from the extinction class are not calculated. Assume 
# for now that it is 1 since the extinction class is not viable. 

###### Absolute Fitness pFix #######
for ii in range(1,nEst):
    # set initial pop sizes
    # eq_Ni indices have to be offset by 1 to match those from di. eq_Ni
    # ranges from indices 0 to 179 for states 1 to 180, and di indices range
    # 0 to 180 for states 0 to 180
    pop = np.asarray([eq_Ni[ii]-1, 1])
    
    # calcualte the d and c's
    d = [di[ii-1],di[ii]]
    c = np.array([1,1])
    
    # estimate pFix values
    pFixAbs_i[ii-1] = myfun.simulation_popEvo_pFixEst(params,pop,d,c,nPfix,fixThrshld)
    
####### Relative Fitness pFix #######    
#for ii in range(1,nEst):    
#    # set initial pop sizes
#    # eq_Ni indices have to be offset by 1 to match those from di. eq_Ni
#    # ranges from indices 0 to 179 for states 1 to 180, and di indices range
#    # 0 to 180 for states 0 to 180
#    pop = np.asarray([eq_Ni[ii]-1, 1])
#    
#    # calcualte the d and c's
#    d = [di[ii],di[ii]]
#    c = np.array([1,(1+params['cr'])])
#    
#    # estimate pFix values
#    pFixRel_i[ii-1] = simulation_popEvo_pFixEst(params,pop,d,c,nPfix,fixThrshld)
#    
## write estimates of pFix values to respective files
#fwrite_abs = open("evoExp01_absPfix.csv","a")
#fwrite_rel = open("evoExp01_absPfix.csv","a") 
#
#for ii in range(len(pFixAbs_i))
#
#
#print(time.time() - t)  


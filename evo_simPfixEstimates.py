# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 2022
@author: kgomez81
"""

import time
from joblib import Parallel, delayed
import numpy as np
import fig_functions as myfun  
import math as math
import csv

import matplotlib.pyplot as plt
import bisect
import scipy as sp

import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy.optimize import root 
import fig_functions as myfun  
import pfix_sim_functions as pfix_sim

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


# begin simulations and time the processes
t = time.time()


element_run_eqpop = Parallel(n_jobs=1)(delayed(myfun.get_eq_pop_density)(parValue[parName['b']],di[k*part],parValue[parName['sa']],yi_option) for k in range(subsamp)) #got rid of the minus 1
print(time.time() - t)  

parallel_eqpop = open("parallel_eqpop.txt","a")
for row in element_run_eqpop:
    parallel_eqpop.write(str(int(math.ceil(parValue[parName['T']]*row[0])))+'\n')
parallel_eqpop.close()

parallel_effsr = open("parallel_effsr.txt","a") 
for row in element_run_eqpop: # b,di,y,cr
    parallel_effsr.write(str(myfun.get_c_selection_coefficient_OLD(parValue[parName['b']],row[0],parValue[parName['cr']]))+'\n')
parallel_effsr.close()


#The idea would be to take this and apply to
d_Inc = 1
c_Inc = 0

t = time.time()
element_run_abs = Parallel(n_jobs=1)(delayed(myfun.modsimpop)(d_Inc,c_Inc,samp,parValue[parName['T']],parValue[parName['cr']],parValue[parName['b']],di[k*part:(k*part + 2)],parValue[parName['do']],((di[k*part]/di[k*part + 1])-1)/(di[k*part + 1]-1),d_max,yi_option) for k in range(subsamp)) #got rid of the minus 1
print(time.time() - t)  

parallel_abs = open("parallel_abs.txt","a")
for row in element_run_abs:
    parallel_abs.write(str(row)+'\n')
parallel_abs.close()

d_Inc = 0
c_Inc = 1
#d_pfixes[i] = myfun.modsimpop(d_Inc,c_Inc,samp,parValue[parName['T']],parValue[parName['cr']],parValue[parName['b']],di[i:(i+2)],parValue[parName['do']],parValue[parName['cr']],d_max,yi_option) #fix
t = time.time()
element_run_rel = Parallel(n_jobs=1)(delayed(myfun.modsimpop)(d_Inc,c_Inc,samp,parValue[parName['T']],parValue[parName['cr']],parValue[parName['b']],di[k*5:(k*5 + 2)],parValue[parName['do']],((di[k*5]/di[k*5 + 1])-1)/(di[k*5 + 1]-1),d_max,yi_option) for k in range(subsamp)) #got rid of the minus 1
print(time.time() - t)  

parallel_rel = open("parallel_rel.txt","a")
for row in element_run_rel:
    parallel_rel.write(str(row)+'\n')
parallel_rel.close()




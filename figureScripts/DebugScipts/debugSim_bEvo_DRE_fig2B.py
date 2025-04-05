# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 01:56:23 2025

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.SimRoutines import SIM_Init as evoInit
from evoLibraries.SimRoutines import SIM_DRE_class as simDre
from evoLibraries.LotteryModel import LM_functions as lmfun

#%% Inestigation attractor and selection coefficients

# debugging effort:
# 1. generate a state space and find the attractor
#    we want to verify selection coefficients at the attractor
# 2. once we have the selection coefficients at the attractor, we are going to 
#    compare them with deterministic dynamics of selection 

############################################
#  Small T Simulation (1E9)
############################################
# filepaths for loading and saving outputs
simPathsIO  = dict()
simPathsIO['paramFile']     = 'evoExp_DRE_bEvo_03A_parameters.csv'
simPathsIO['paramTag']      = 'param_03A_DRE_bEvo'
simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2B'
simPathsIO['statsFile']     = 'sim_Fig2B_T1E9_stats'
simPathsIO['snpshtFile']    = 'sim_Fig2B_T1E9_snpsht'
simPathsIO['fullStochModelFlag']    = False
simPathsIO['simpleEnvShift']        = True
simPathsIO['modelType']             = 'DRE'
simPathsIO['absFitType']            = 'bEvo'

# specify parameters for the MC models
simArryData = dict()
simArryData['tmax'] = 5 # 10 generations
simArryData['tcap'] = 5
simArryData['nij']          = np.array([[3e8,0],[1,0]])
simArryData['bij_mutCnt']   = np.array([[27,27],[28,28]])
simArryData['dij_mutCnt']   = np.array([[1,1],[1,1]])  
simArryData['cij_mutCnt']   = np.array([[1,2],[1,2]])

# group sim parameters
simInit = evoInit.SimEvoInit(simPathsIO,simArryData)

# generate sim object and run to get some of the outputs
evoSim = simDre.simDREClass(simInit)
evoSim.run_evolutionModel()


outpar = ['C:\\Users\\dirge\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\test_selection.csv',1]

tmax = 10
T = simInit.mcModel.params['T']

lmfun.run_lotteryModelSelection_noMutations(evoSim.nij,evoSim.get_bij(),evoSim.get_dij(),evoSim.get_cij(),1e9,tmax,outpar)


#%% test selection 

sampleName = ''
# evoSnapshotFile = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250321_1516.pickle'
evoSnapshotFile = 'C:\\Users\\dirge\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_'+sampleName+'.pickle'
with open(evoSnapshotFile, 'rb') as file:
    # Serialize and write the variable to the file
    loaded_data = pickle.load(file)
    
evoSimTest1 = simDre.simDREClass(loaded_data)

# factorUT = (1-np.sum(evoSimTest.nij)/evoSimTest.mcModel.params['T'])
# evoSimTest.mcModel.params['Ua']*evoSimTest.nij*evoSimTest.get_bij()*factorUT

idxEq = evoSimTest1.mcModel.get_mc_stable_state_idx()

# Ni  = evoSimTest1.mcModel.eq_Ni[idxEq]
# Uai = evoSimTest1.mcModel.params['Ua']
# sai = evoSimTest1.mcModel.sc_i[idxEq]
# Uci = evoSimTest1.mcModel.params['Uc']
# sci = evoSimTest1.mcModel.sc_i[idxEq]

# Test = 1/(Ni*si*Ui)
# Tswp = (1/si)*np.log(Ni*si)

# va = sai**2*(2*np.log(Ni*sai)-np.log(sai/Uai))/(np.log(sai/Uai))**2
# vc = sci**2*(2*np.log(Ni*sci)-np.log(sci/Uci))/(np.log(sci/Uci))**2

# qa = 2*np.log(Ni*sai)/np.log(sai/Uai)
# qc = 2*np.log(Ni*sci)/np.log(sci/Uci)


plt.imshow(np.log10(evoSimTest1.nij))
plt.colorbar()
plt.show()



fig,ax = plt.subplots(1,1)
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.va_i,c='blue')
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.vc_i,c='red')
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.ve_i,c='black')


#%% 
evoSnapshotFile = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\sim_Fig2B_T1E12_snpsht_param_03B_DRE_bEvo_20250321_1517.pickle'
with open(evoSnapshotFile, 'rb') as file:
    # Serialize and write the variable to the file
    loaded_data = pickle.load(file)
    
evoSimTest2 = simDre.simDREClass(loaded_data)

# factorUT = (1-np.sum(evoSimTest.nij)/evoSimTest.mcModel.params['T'])
# evoSimTest.mcModel.params['Ua']*evoSimTest.nij*evoSimTest.get_bij()*factorUT
idxEq = evoSimTest2.mcModel.get_mc_stable_state_idx()

# Ni = evoSimTest.mcModel.eq_Ni[idxEq]
# Uai = evoSimTest.mcModel.params['Ua']
# sai = evoSimTest.mcModel.sc_i[idxEq]
# Uci = evoSimTest.mcModel.params['Uc']
# sci = evoSimTest.mcModel.sc_i[idxEq]

# Test = 1/(Ni*si*Ui)
# Tswp = (1/si)*np.log(Ni*si)

# va = sai**2*(2*np.log(Ni*sai)-np.log(sai/Uai))/(np.log(sai/Uai))**2
# vc = sci**2*(2*np.log(Ni*sci)-np.log(sci/Uci))/(np.log(sci/Uci))**2

# qa = 2*np.log(Ni*sai)/np.log(sai/Uai)
# qc = 2*np.log(Ni*sci)/np.log(sci/Uci)

fig,ax = plt.subplots(1,1)
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.va_i,c='blue')
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.vc_i,c='red')
ax.plot(evoSimTest2.mcModel.state_i,evoSimTest2.mcModel.va_i,c='cyan')
ax.plot(evoSimTest2.mcModel.state_i,evoSimTest2.mcModel.vc_i,c='magenta')

evoSimTest1.mcModel.get_mc_stable_state_idx()
evoSimTest2.mcModel.get_mc_stable_state_idx()



# 2. burn in up to attractor with 10 generations



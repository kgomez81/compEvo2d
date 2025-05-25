# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:42:16 2025

@author: Owner
"""

import numpy as np
import pickle

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.SimRoutines import SIM_Init as evoInit
from evoLibraries.SimRoutines import SIM_DRE_class as simDre
import figFunctions as figfun

#%%

# debugging effort:
# 1. generate a state space and find the attractor
#    we want to verify selection coefficients at the attractor
# 2. once we have the selection coefficients at the attractor, we are going to 
#    compare them with deterministic dynamics of selection 

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

############################################
#  Small T Simulation (1E9)
# 
#  Normal run to check vb
#
############################################
# filepaths for loading and saving outputs

simPathsIO  = dict()
simArryData = dict()

# specify parameters for input / output paths
simPathsIO['paramFile']     = 'evoExp_DRE_bEvo_03A_parameters.csv'
simPathsIO['paramTag']      = 'param_03A_DRE_bEvo'

simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2B_MutOnly'
simPathsIO['statsFile']     = 'sim_Fig2B_T1E9_stats'
simPathsIO['snpshtFile']    = 'sim_Fig2B_T1E9_snpsht'

simPathsIO['modelDynamics']         = 2
simPathsIO['simpleEnvShift']        = True
simPathsIO['modelType']             = 'DRE'
simPathsIO['absFitType']            = 'bEvo'

# specify parameters for the MC models
simArryData['tmax'] = 1000
simArryData['tcap'] = 1

simArryData['nij']          = np.array([[3e8]])
simArryData['bij_mutCnt']   = np.array([[16]])
simArryData['dij_mutCnt']   = np.array([[1]])  
simArryData['cij_mutCnt']   = np.array([[1]])

# group sim parameters
simInit = evoInit.SimEvoInit(simPathsIO,simArryData)

# setting poulation size to equilibrium value
simInit.nij[0,0] = simInit.mcModel.eq_Ni[int(simInit.bij_mutCnt[0,0])]

# generate sim object and run
evoSim = simDre.simDREClass(simInit)
evoSim.run_evolutionModel()

figfun.plot_simulationAnalysis(evoSim)

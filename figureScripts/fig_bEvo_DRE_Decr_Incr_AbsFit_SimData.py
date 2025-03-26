"""
Created on Fri Feb 25 12:39:02 2022

@author: dirge
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
Simulation script to generate data for manuscript Figure 2B
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import numpy as np
import pickle

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.SimRoutines import SIM_Init as evoInit
from evoLibraries.SimRoutines import SIM_DRE_class as simDre

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

############################################
#  Small T Simulation (1E7)
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
simArryData['tmax'] = 5000
simArryData['tcap'] = 50

simArryData['nij']          = np.array([[3e8]])
simArryData['bij_mutCnt']   = np.array([[28]])
simArryData['dij_mutCnt']   = np.array([[1]])  
simArryData['cij_mutCnt']   = np.array([[1]])

# group sim parameters
simInit = evoInit.SimEvoInit(simPathsIO,simArryData)

# generate sim object and run
evoSim = simDre.simDREClass(simInit)
evoSim.run_evolutionModel()


# ############################################
# #  large T Simulation (1E12)
# ############################################
# # filepaths for loading and saving outputs
# simPathsIO  = dict()

# simPathsIO['paramFile']     = 'evoExp_DRE_bEvo_03B_parameters.csv'
# simPathsIO['paramTag']      = 'param_03B_DRE_bEvo'

# simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2B'
# simPathsIO['statsFile']     = 'sim_Fig2B_T1E12_stats'
# simPathsIO['snpshtFile']    = 'sim_Fig2B_T1E12_snpsht'

# simPathsIO['fullStochModelFlag']    = False
# simPathsIO['simpleEnvShift']        = True
# simPathsIO['modelType']             = 'DRE'
# simPathsIO['absFitType']            = 'bEvo'

# # specify parameters for the MC models
# simArryData = dict()
# simArryData['tmax'] = 10 #1000000
# simArryData['tcap'] = 1 #50

# simArryData['nij']          = np.array([[1e11]])
# simArryData['bij_mutCnt']   = np.array([[20]])
# simArryData['dij_mutCnt']   = np.array([[1]])  
# simArryData['cij_mutCnt']   = np.array([[1]])

# # group sim parameters
# simInit = evoInit.SimEvoInit(simPathsIO,simArryData)

# # generate sim object and run
# evoSim = simDre.simDREClass(simInit)
# evoSim.run_evolutionModel()

#%%
######################
# TEST Load
######################

# evoSnapshotFile = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\sim_Fig2B_T1E7_snpsht_param_04A_DRE_bEvo_20250317_2314.pickle'
# with open(evoSnapshotFile, 'rb') as file:
#     # Serialize and write the variable to the file
#     loaded_data = pickle.load(file)
    
# evoSimTest = simDre.simDREClass(loaded_data)
# evoSimTest.run_evolutionModel()

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

import figFunctions as figfun


#%% ------------------------------------------------------------------------
# simulations
# --------------------------------------------------------------------------

def runSimulation_small_T():
    
    ############################################
    #  Small T Simulation (1E7)
    ############################################
    # filepaths for loading and saving outputs
    simPathsIO  = dict()
    
    simPathsIO['paramFile']     = 'evoExp_DRE_bEvo_03A_parameters.csv'
    simPathsIO['paramTag']      = 'param_03A_DRE_bEvo'
    
    simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2B_FINAL'
    simPathsIO['statsFile']     = 'sim_Fig2B_T1E9_stats'
    simPathsIO['snpshtFile']    = 'sim_Fig2B_T1E9_snpsht'
    
    simPathsIO['modelDynamics']         = 2
    simPathsIO['simpleEnvShift']        = True
    simPathsIO['modelType']             = 'DRE'
    simPathsIO['absFitType']            = 'bEvo'
    
    # specify parameters for the MC models
    simArryData = dict()
    simArryData['tmax'] = 100000
    simArryData['tcap'] = 5
    
    simArryData['nij']          = np.array([[3e8]])
    simArryData['bij_mutCnt']   = np.array([[28]])
    simArryData['dij_mutCnt']   = np.array([[1]])  
    simArryData['cij_mutCnt']   = np.array([[1]])
    
    # group sim parameters
    simInit = evoInit.SimEvoInit(simPathsIO,simArryData)
    
    # setting poulation size to equilibrium value, and set a non-zero rate of
    # environmental change. We'll need to recalculate the MC model for the latter
    # so that vE is is correct in the MC model.
    simInit.params['se'] = 0.05
    simInit.nij[0,0] = simInit.mcModel.eq_Ni[int(simInit.bij_mutCnt[0,0])]
    
    # recalculate MC model with change to rate of environmental change and 
    # turn off mutations to analyze just the dynamics from environmental changes
    simInit.recaculate_mcModel()
    simInit.turn_off_deleterious_mutations()
    
    # generate sim object and run
    evoSim = simDre.simDREClass(simInit)
    evoSim.run_evolutionModel()
    
    figfun.plot_simulationAnalysis(evoSim)
    
    # save evoSim
    evoSimFile = evoSim.outputSnapshotFile.replace('.pickle','_evoSim.pickle')
    with open(evoSimFile, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(evoSim, file)

    return None

def runSimulation_large_T():
    ############################################
    #  large T Simulation (1E12)
    ############################################
    # filepaths for loading and saving outputs
    simPathsIO  = dict()
    
    simPathsIO['paramFile']     = 'evoExp_DRE_bEvo_03B_parameters.csv'
    simPathsIO['paramTag']      = 'param_03B_DRE_bEvo'
    
    simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2B_FINAL'
    simPathsIO['statsFile']     = 'sim_Fig2B_T1E12_stats'
    simPathsIO['snpshtFile']    = 'sim_Fig2B_T1E12_snpsht'
    
    simPathsIO['modelDynamics']         = 2
    simPathsIO['simpleEnvShift']        = True
    simPathsIO['modelType']             = 'DRE'
    simPathsIO['absFitType']            = 'bEvo'
    
    # specify parameters for the MC models
    simArryData = dict()
    simArryData['tmax'] = 100000
    simArryData['tcap'] = 5 
    
    simArryData['nij']          = np.array([[1e11]])
    simArryData['bij_mutCnt']   = np.array([[28]])
    simArryData['dij_mutCnt']   = np.array([[1]])  
    simArryData['cij_mutCnt']   = np.array([[1]])
    
    # group sim parameters
    simInit = evoInit.SimEvoInit(simPathsIO,simArryData)
    
    # setting poulation size to equilibrium value, and set a non-zero rate of
    # environmental change. We'll need to recalculate the MC model for the latter
    # so that vE is is correct in the MC model.
    simInit.params['se'] = 0.05
    simInit.nij[0,0] = simInit.mcModel.eq_Ni[int(simInit.bij_mutCnt[0,0])]
    
    # recalculate MC model with change to rate of environmental change and 
    # turn off mutations to analyze just the dynamics from environmental changes
    simInit.recaculate_mcModel()
    simInit.turn_off_deleterious_mutations()
    
    # generate sim object and run
    evoSim = simDre.simDREClass(simInit)
    evoSim.run_evolutionModel()
    
    figfun.plot_simulationAnalysis(evoSim)
    
    # save evoSim
    evoSimFile = evoSim.outputSnapshotFile.replace('.pickle','_evoSim.pickle')
    with open(evoSimFile, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(evoSim, file)
        
    return None

def main():
    
    # run small T simulation
    runSimulation_small_T()
    
    # run large T simulation
    runSimulation_large_T()
    
    
if __name__ == "__main__":
    main()
    


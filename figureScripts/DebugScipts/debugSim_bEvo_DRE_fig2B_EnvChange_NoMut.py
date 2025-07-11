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
# 1. Run a simulation with environmental changes only, and measure the rate of fitness decrease
# NOTE: copy this script to figureScripts folder directory above, and run from that location 

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

def runSimulation():
    
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
    
    simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2B_EnvChange'
    simPathsIO['statsFile']     = 'sim_Fig2B_T1E9_stats'
    simPathsIO['snpshtFile']    = 'sim_Fig2B_T1E9_snpsht'
    
    simPathsIO['modelDynamics']         = 1
    simPathsIO['simpleEnvShift']        = True
    simPathsIO['modelType']             = 'DRE'
    simPathsIO['absFitType']            = 'bEvo'
    
    # specify parameters for the MC models
    simArryData['tmax'] = 2000
    simArryData['tcap'] = 1     # note: capture each iteration to est. vE
    
    simArryData['nij']          = np.array([[3e8]])
    simArryData['bij_mutCnt']   = np.array([[60]])
    simArryData['dij_mutCnt']   = np.array([[1]])  
    simArryData['cij_mutCnt']   = np.array([[1]])
    
    # group sim parameters
    simInit = evoInit.SimEvoInit(simPathsIO,simArryData)
    
    # setting poulation size to equilibrium value, and set a non-zero rate of
    # environmental change. We'll need to recalculate the MC model for the latter
    # so that vE is is correct in the MC model.
    simInit.params['se'] = 0.06
    simInit.nij[0,0] = simInit.mcModel.eq_Ni[int(simInit.bij_mutCnt[0,0])]
    
    # recalculate MC model with change to rate of environmental change and 
    # turn off mutations to analyze just the dynamics from environmental changes
    simInit.recaculate_mcModel()
    simInit.turn_off_mutations()
    
    # generate sim object and run
    evoSim = simDre.simDREClass(simInit)
    evoSim.run_evolutionModel()
    
    # plot the results for measuring the rate of environmental changes
    figfun.plot_environmentalChange(evoSim)
    
def main():
    np.random.seed(2)
    
    # run simulation
    runSimulation()
    
if __name__ == "__main__":
    main()

"""
Created on Fri Feb 25 12:39:02 2022

@author: dirge
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
Simulation script to generate data for manuscript Figure 1A/1B.
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

# imports to run the simulation
from evoLibraries.SimRoutines import SIM_Init as evoInit
from evoLibraries.SimRoutines import SIM_DRE_class as simDre

# helper functions to manage data and generate plots
import figFunctions as figfun

#%% ------------------------------------------------------------------------
# methods to setup and execute simulation runs for figures 2A/B
# --------------------------------------------------------------------------

def getSimInit(init):
    # The method getSimInit() takes in arguments to specify the struct needed
    # to initialize a simulation object.
    simPathsIO  = dict()
    
    # I/O settings
    simPathsIO['paramFile']     = init['paramFile']
    simPathsIO['paramTag']      = init['paramTag']
    
    simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig1'
    simPathsIO['statsFile']     = init['statsFile']
    simPathsIO['snpshtFile']    = init['snpshtFile']
    
    simPathsIO['modelDynamics']         = 2
    simPathsIO['simpleEnvShift']        = True
    simPathsIO['modelType']             = 'DRE'
    simPathsIO['absFitType']            = 'bEvo'
    
    # specify parameters for the MC models
    # 1. tmax = maximum iterations of sim run
    # 2. tcap = capture snapshots of states each generation (5 iterations)
    simArryData = dict()
    simArryData['tmax'] = init['tmax']
    simArryData['tcap'] = 5
    
    simArryData['nij']          = np.array([[3e8]])
    simArryData['bij_mutCnt']   = np.array([[init['initState']]])
    simArryData['dij_mutCnt']   = np.array([[1]])  
    simArryData['cij_mutCnt']   = np.array([[1]])
    
    # group sim parameters
    simInit = evoInit.SimEvoInit(simPathsIO,simArryData)
    
    # setting poulation size to equilibrium value, and set a non-zero rate of
    # environmental change. We'll need to recalculate the MC model for the latter
    # so that vE is is correct in the MC model.
    simInit.nij[0,0] = simInit.mcModel.eq_Ni[int(simInit.bij_mutCnt[0,0])]
    
    # recalculate MC model with changes to params above
    simInit.recaculate_mcModel()
    
    return simInit

def runSimulation(simInit):
    # runSimulation() takes initialization parameters and creates the sim
    # object needed to execute a simulation run.
    
    # generate sim object and run
    evoSim = simDre.simDREClass(simInit)
    evoSim.run_evolutionModel()
    
    # save evoSim
    figfun.save_evoSnapshot(evoSim)
    
    # generate plots for stats
    figfun.plot_simulationAnalysis(evoSim)
    figfun.plot_mcModel_histAndVaEstimates(evoSim)
        
    return None

def main():
    # main method to run the various simulation
        
    # define input file paths 
    # 
    parFiles = ['01','02']
    parTag   = ['EnvLim','ComLim']
    figPanel = ['A','B']
    start_i  = [50,100]
    t_stop   = [5E6,10E6]

    # dictionary for inputs
    init = dict()
    
    for ii in range(len(parFiles)):
        # set up sim initialization items that vary
        init['paramFile']  = ('evoExp_DRE_bEvo_%s_parameters.csv' % (parFiles[ii]))
        init['paramTag']   = ('param_%s_DRE_bEvo_%s' % (parFiles[ii],parTag[ii]))
        init['statsFile']  = ('sim_Fig2%s_%s_stats' % (figPanel[ii],parTag[ii]))
        init['snpshtFile'] = ('sim_Fig2%s_%s_snpsht' % (figPanel[ii],parTag[ii]))
        init['initState']  = start_i[ii]
        init['tmax']       = t_stop[ii]
        
        # run small T simulation
        simInit_ii = getSimInit(init)
    
        # run large T simulation
        runSimulation(simInit_ii)
    
if __name__ == "__main__":
    main()
    


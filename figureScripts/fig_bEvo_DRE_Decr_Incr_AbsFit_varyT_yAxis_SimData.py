"""
Created on Fri Feb 25 12:39:02 2022

@author: dirge
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
Simulation script to generate data for manuscript Figure 2A/B.
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
    
    simPathsIO['simDatDir']     = 'sim_bEvo_DRE_Fig2'
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
    simInit.params['se'] = init['se_size']
    simInit.params['T']  = init['terrSize']
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
    parFiles = ['03','03','04','04']
    terrSize = ['1E9','1E12','1E9','1E12']
    figPanel = ['A','A','B','B']
    start_i  = [28,28,28,28]
    se_size  = [0.06,0.06,0.06,0.06]
    t_stop   = [5E4,5E4,1E6,5E4]

    # dictionary for inputs
    init = dict()
    
    # for ii in range(len(parFiles)):
    for ii in [2]:    
        # set up sim initialization items that vary
        init['paramFile']  = ('evoExp_DRE_bEvo_%s_parameters.csv' % (parFiles[ii]))
        init['paramTag']   = ('param_%s_DRE_bEvo_T%s' % (parFiles[ii],terrSize[ii]))
        init['statsFile']  = ('sim_Fig2%s_T%s_stats' % (figPanel[ii],terrSize[ii]))
        init['snpshtFile'] = ('sim_Fig2%s_T%s_snpsht' % (figPanel[ii],terrSize[ii]))
        init['terrSize']   = float(terrSize[ii])
        init['initState']  = start_i[ii]
        init['se_size']    = se_size[ii]
        init['tmax']       = t_stop[ii]
        
        # run small T simulation
        simInit_ii = getSimInit(init)
    
        # run large T simulation
        runSimulation(simInit_ii)
    
if __name__ == "__main__":
    main()
    


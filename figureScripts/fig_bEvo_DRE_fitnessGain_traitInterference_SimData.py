# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab

Simulation script to generate data for manuscript figure characterizing degree
of stalling across different Rho values.

We produce three different 2 different data sets for 4 manuscript figures. The
data sets themselves will require multiple runs to sample the various ve values
that can be considered to characterize the dependence of the intersection on 
the rate of environmental change.

1. simulation runs based on parameters of figure 2A of the manuscript, which 
   represent a rho > 1 setting
   
2. simulation runs based on parameters of figure 2B of the manuscript, which 
   represent a rho < 1 setting
   
Some key modifcations to the files are as follows
- turn off parallel computations for the MC state space
- use method 2 for calculating pfix values across the state space
- adjustments to environmental parameters to fascilitate sweeps of ve/(vc=va) 
   
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

import time

# import libraries for parallelization
from joblib import Parallel, delayed, cpu_count

# imports to run the simulation
from evoLibraries.SimRoutines import SIM_Init as evoInit
from evoLibraries.SimRoutines import SIM_DRE_class as simDre

# helper functions to manage data and generate plots
import figFunctions as figfun

#%% ------------------------------------------------------------------------
# methods to setup and execute simulation runs for figures 2A/B
# --------------------------------------------------------------------------

def getSimInit(init,outputDir):
    # The method getSimInit() takes in arguments to specify the struct needed
    # to initialize a simulation object.
    simPathsIO  = dict()
    
    # I/O settings
    simPathsIO['paramFile']     = init['paramFile']
    simPathsIO['paramTag']      = init['paramTag']
    
    simPathsIO['simDatDir']     = outputDir             # Eg. 'sim_bEvo_DRE_VeFitChng_NewTest'
    simPathsIO['statsFile']     = init['statsFile']
    simPathsIO['snpshtFile']    = init['snpshtFile']
    
    # --------------------------------------------------------------------------
    # Setup of parameters and MC model
    # --------------------------------------------------------------------------
    # Some key definitions: 
    #  - modelDyanmics - flag to indicate the expected type of model dynamics for
    #                    for mutation. Either:
    #                    0: full stochastics dynamics (no implemented)
    #                    1: detrministic model with/without mutations, and 
    #                    environmental changes.
    #                    !{0,1}: lottery model of selection with Poisson sampling
    #
    #  - simpleEnvShft - Either: 
    #                    1) for simple shifts back with rates scaled to achieve 
    #                    desired rate of environmental degredation (variable R), or 
    #                    2) for shifts back that shuffle abudances to decrease
    #                    fitness of individiuals by fixed amount (fixed s)
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
    
    # setting poulation size to equilibrium value, and 
    simInit.nij[0,0]    = simInit.mcModel.eq_Ni[int(simInit.bij_mutCnt[0,0])]
    
    # set a rate of environmental change as fraction of vc=va intersection. 
    # do this by finding va=vc value, then calculating a new se using
    #
    #    ve = se*R*tau = fraction * va(=vc) ==> se = fraction * va / (R * tau)
    #
    # where se is fitness decline from env, R is rate of env events per model 
    # iteration, and tau is model iterations per generation.
    #
    inters_idx              = simInit.mcModel.get_va_vc_intersection_index()[0]
    va_intersect            = simInit.mcModel.va_i[int(inters_idx)]
    tau_intersect           = 1/(simInit.mcModel.di[int(inters_idx)]-1)
    v_fraction              = init['veSize']/100.0
    T_fraction              = init['TSize']/100.0
    
    # implement the equation above
    simInit.params['se']    = v_fraction * va_intersect / (simInit.params['R'] * tau_intersect)
    
    # adjust Territory size
    simInit.params['T']     = T_fraction * simInit.params['T']
    
    # recalculate MC model with changes to params above
    simInit.recaculate_mcModel()
    
    return simInit

# --------------------------------------------------------------------------

def get_simInit(paramDefs,ii,outputDir):
    # simple function to generate init structs for simulation runs simulation 
    # paramDefs should be a dictionary with lists entries for 
    #
    # parFiles:  list of strings to select an input file
    # figPanel:  list of strings indicate the panel it belongs to for figure 4 (manuscript)
    # veSamples: list of ve/(vc=va) percentages to define the rate of adaptation
    # start_i:   list of states to initialize the simulation
    # t_stop:    list of stopping times for the simulation runs
    
    # dictionary for inputs
    init = dict()
    
    # set up sim initialization items that vary
    init['paramFile']  = ('evoExp_DRE_bEvo_%s_parameters.csv'   % (paramDefs['paramFile'][ii]))
    init['paramTag']   = ('param_%s_DRE_bEvo'                   % (paramDefs['paramFile'][ii]))
    init['statsFile']  = ('sim_Fig%s%s_ve%s_T%s_stats'          % (paramDefs['figNumber'][ii],paramDefs['figPanel'][ii],paramDefs['veSize'][ii],paramDefs['TSize'][ii]))
    init['snpshtFile'] = ('sim_Fig%s%s_ve%s_T%s_snpsht'         % (paramDefs['figNumber'][ii],paramDefs['figPanel'][ii],paramDefs['veSize'][ii],paramDefs['TSize'][ii]))
    init['initState']  = paramDefs['start_i'][ii]
    init['tmax']       = paramDefs['t_stop'][ii]
    
    # set ve fraction of vc=va, and T-fraction of paramfile T
    init['veSize']     = float(paramDefs['veSize'][ii])
    init['TSize']      = float(paramDefs['TSize'][ii])
        
    return getSimInit(init,outputDir)

# --------------------------------------------------------------------------

def write_outputfile_list(outputfiles,save_name):
    # small function to output the list of sim runs from a run set into the 
    # output directory
    
    # set the full path to save the run details
    full_save_name = os.path.join(outputfiles[0][-1],save_name)
    
    # headers
    headers = ['fig_panel','ve_percent','T_percent','sim_runtime','sim_stats','adap_log_abs','adap_log_rel','sim_snapshot','output_dir']
    
    # loop through list of output files
    for ii in range(len(outputfiles)):
        
        # open the file and append new data
        with open(full_save_name, "a") as file:
            if (ii==0):
                # output column if at initial time
                file.write( ','.join(tuple(headers))+'\n' )
            
            # output data collected
            file.write( (','.join(tuple(['%s']*len(outputfiles[ii]))) + '\n') % tuple(outputfiles[ii]))
    
    return None

# --------------------------------------------------------------------------

def runSimulation(simInit):
    # runSimulation() takes initialization parameters and creates the sim
    # object needed to execute a simulation run.
    
    # generate sim object and run
    evoSim = simDre.simDREClass(simInit)
    
    # run the simulation
    sim_runtime = time.time()
    evoSim.run_evolutionModel()
    sim_runtime = (time.time() - sim_runtime)/3600.0  # hrs
    
    # save evoSim
    figfun.save_evoSnapshot(evoSim)
    
    # # generate plots for stats (takes too long to produce)
    # figfun.plot_simulationAnalysis(evoSim)
    # figfun.plot_mcModel_histAndVaEstimates(evoSim)
    
    # return collections of the relevant output files to use for figure generation
    sim_run_output_files = []
    
    # save run time
    sim_run_output_files.append(str(sim_runtime))
    
    # file with times and mean fitness statistics
    sim_run_output_files.append(evoSim.outputStatsFile)
    
    # log of adative events to estimate rates of adaptation
    sim_run_output_files.append(evoSim.get_adaptiveEventsLogFilename('abs'))
    sim_run_output_files.append(evoSim.get_adaptiveEventsLogFilename('rel'))
    
    # pickle snapshot of evo object, needed to build plots
    sim_run_output_files.append(evoSim.get_evoSimFilename())
    
    # save output directory
    sim_run_output_files.append(os.path.split(evoSim.get_evoSimFilename())[0])
    
    print("Completed run: %s\nSim runtime: %.3f hrs\n\n" % (os.path.split(evoSim.get_evoSimFilename())[1],sim_runtime))
    
    return sim_run_output_files

# --------------------------------------------------------------------------

def param_set_varyVe():
    # param_set_varyVe parameter sets 
    
    # dictionary to setup parameters
    paramDefs = dict()
    
    # define input file paths setups
    paramDefs['paramFile'] = []    # parameter file to use
    paramDefs['figNumber'] = []    # intended fig number
    paramDefs['figPanel' ] = []    # intended fig panel 
    paramDefs['veSize'   ] = []    # percent of vc=va value
    paramDefs['TSize'    ] = []    # percent of T in param file
    paramDefs['start_i'  ] = []    # starting state
    paramDefs['t_stop'   ] = []    # sim stopping time
    
    # max sim time
    tMax = 3E7
    
    # ve sizes
    vef  = [str(int(20*kk+10)) for kk in range(0,5)]
    nPar = len(vef)
    
    # add the full set of simulations for the new figures
    # each set will include sampled simulations from 10,20,...,100
    
    ##############################
    # Rho < 1 sample set    
    ##############################
    
    # define input file paths setups
    paramDefs['paramFile']  .extend(['04B'      for kk in range(nPar)])
    paramDefs['figNumber']  .extend(['4'        for kk in range(nPar)])
    paramDefs['figPanel']   .extend(['A'        for kk in range(nPar)])
    paramDefs['veSize']     .extend([vef[kk]    for kk in range(nPar)])
    paramDefs['start_i']    .extend([95         for kk in range(nPar)])
    paramDefs['t_stop']     .extend([tMax       for kk in range(nPar)])
    paramDefs['TSize']      .extend([str(100)   for kk in range(nPar)])

    ##############################
    # Rho ~ 1 sample set
    ##############################
    # These runs take much longer because rho > 1 parameter sets 
    # often have low va conditions, so significantly more iterations
    # are needed to get the same number of sample sojourn times 
    
    # define input file paths setups
    paramDefs['paramFile']  .extend(['04A'      for kk in range(nPar)])
    paramDefs['figNumber']  .extend(['4'        for kk in range(nPar)])
    paramDefs['figPanel']   .extend(['B'        for kk in range(nPar)])
    paramDefs['veSize']     .extend([vef[kk]    for kk in range(nPar)])
    paramDefs['start_i']    .extend([145        for kk in range(nPar)])
    paramDefs['t_stop']     .extend([tMax       for kk in range(nPar)])
    paramDefs['TSize']      .extend([str(100)   for kk in range(nPar)])
    
    ##############################
    # Rho > 1 sample set
    ##############################
    # These runs take much longer because rho > 1 parameter sets 
    # often have low va conditions, so significantly more iterations
    # are needed to get the same number of sample sojourn times 
    
    # define input file paths setups
    paramDefs['paramFile']  .extend(['03A'      for kk in range(nPar)])
    paramDefs['figNumber']  .extend(['4'        for kk in range(nPar)])
    paramDefs['figPanel']   .extend(['C'        for kk in range(nPar)])
    paramDefs['veSize']     .extend([vef[kk]    for kk in range(nPar)])
    paramDefs['start_i']    .extend([75         for kk in range(nPar)])
    paramDefs['t_stop']     .extend([tMax       for kk in range(nPar)])
    paramDefs['TSize']      .extend([str(100)   for kk in range(nPar)])
     
    return paramDefs

# --------------------------------------------------------------------------

def param_set_varyT():
    
    # dictionary to setup parameters
    paramDefs = dict()
    
    # define input file paths setups
    paramDefs['paramFile'] = []    # parameter file to use
    paramDefs['figNumber'] = []    # intended fig number
    paramDefs['figPanel' ] = []    # intended fig panel 
    paramDefs['veSize'   ] = []    # percent of vc=va value
    paramDefs['TSize'    ] = []    # percent of T in param file
    paramDefs['start_i'  ] = []    # starting state
    paramDefs['t_stop'   ] = []    # sim stopping time
    
    # max sim time
    tMax = 1E7
    
    # add the full set of simulations for the new figures with sampling design
    # each set will include sampled simulations from 10,20,...,100
    vef = [50, 75, 100]
    Ttf = [100, 500, 10000]
    
    kkMap = []
    for ii in range(len(vef)):
        for jj in range(len(Ttf)):
            kkMap.extend([[ii,jj]])

    nMap = len(kkMap)

    ##############################
    # Rho ~ 1 sample set
    ##############################
    # These runs take much longer because rho > 1 parameter sets 
    # often have low va conditions, so significantly more iterations
    # are needed to get the same number of sample sojourn times 
    
    # define input file paths setups
    paramDefs['paramFile']  .extend(['04A'                  for kk in range(nMap)])
    paramDefs['figNumber']  .extend(['5'                    for kk in range(nMap)])
    paramDefs['figPanel']   .extend(['B'                    for kk in range(nMap)])
    paramDefs['veSize']     .extend([str(vef[kkMap[kk][0]]) for kk in range(nMap)])
    paramDefs['start_i']    .extend([145                    for kk in range(nMap)])
    paramDefs['t_stop']     .extend([tMax                   for kk in range(nMap)])
    paramDefs['TSize']      .extend([str(Ttf[kkMap[kk][1]]) for kk in range(nMap)])
     
    return paramDefs

# --------------------------------------------------------------------------

def main():
    # Main method to run the various simulation. We parallize across sim runs, 
    # which means we cannot use parallelization across the MC system state space
    # or parallelization across pfix calculations.
    #
    # For parallelization, a main function to use in the following way
    #
    # arrayResults = Parallel(n_jobs=cpu_count())
    #           ( delayed(_myFunction) ( tuple_params(kk) ) for kk in range(arraySize) )

    # ###################################################
    # #### PART I - ve variation and intersections ######
    # ###################################################
    
    # # dictionary to setup parameters for runs with ve 
    # paramDefs   = param_set_varyVe()
    # nSims       = len(paramDefs['paramFile'])
    # outputDir   = 'sim_bEvo_DRE_VeFitChng_NewTest'
    
    # # carry out sim runs in parallel
    # outputfiles = Parallel(n_jobs=cpu_count()-1)(delayed(runSimulation)(get_simInit(paramDefs,kk,outputDir)) for kk in range(nSims))
    
    # # non parallel verion
    # # outputfiles = [runSimulation(get_simInit(paramDefs,kk)) for kk in range(1)]
    
    # # add to the list the selected ve size
    # for ii in range(nSims):
    #     outputfiles[ii] = [paramDefs['figPanel'][ii],paramDefs['veSize'][ii],paramDefs['TSize'][ii]] + outputfiles[ii]
    
    # # save a list of the output files in the output directory
    # save_name = 'simList_bEvo_DRE_fitnessGain_traitInterference_veFitChng.csv'
    # write_outputfile_list(outputfiles,save_name)
    
    ###################################################
    #### PART II - T variation and intersections ######
    ###################################################
    
    # dictionary to setup parameters for runs with ve 
    paramDefs   = param_set_varyT()
    nSims       = len(paramDefs['paramFile'])
    outputDir   = 'sim_bEvo_DRE_TFitChng_NewTest'
    
    # carry out sim runs in parallel
    outputfiles = Parallel(n_jobs=cpu_count()-1)(delayed(runSimulation)(get_simInit(paramDefs,kk,outputDir)) for kk in range(nSims))
    
    # non parallel verion
    # outputfiles = [runSimulation(get_simInit(paramDefs,kk)) for kk in range(1)]
    
    # add to the list the selected ve size
    for ii in range(nSims):
        outputfiles[ii] = [paramDefs['figPanel'][ii],paramDefs['veSize'][ii],paramDefs['TSize'][ii]] + outputfiles[ii]
    
    # save a list of the output files in the output directory
    save_name = 'simList_bEvo_DRE_fitnessGain_traitInterference_TFitChng.csv'
    write_outputfile_list(outputfiles,save_name)
    
if __name__ == "__main__":
    main()
    


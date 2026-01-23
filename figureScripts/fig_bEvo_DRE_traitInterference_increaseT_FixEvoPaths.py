# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: Kevin Gomez
Script to generate plots that show fitness gains across changes in T. These 
are compared with rho estimates.
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import pandas as pd
import numpy as np

import time
import pickle 

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

# helper functions to manage data and generate plots
from evoLibraries.LotteryModel import LM_functions as lmfun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

def fix_evoFiles(figSetup,panelkey):
    # get_figData() loops throught the various panel data files and calculates
    # the required figure data. It requires a dictionary with the output
    # directory and filenames where data is store for a simulation run.
    
    # get list of files which are saved to the csv with all of the sim runs
    # for a group of simulations runs of this figure set.
    fileList = os.path.join(figSetup['workDir'],figSetup['dataList'])
    dataFiles = pd.read_csv(fileList)
    dataFiles = dataFiles[dataFiles['fig_panel']==panelkey]
    
    # Number of runs in current sim run set, iterating percentage changes in T,
    # with different starting percentages of vE.
    nFiles = len(dataFiles)
    
        
    # Now we loop through each file, get the data sets for entry 1 and entry 2.
    # However, we note that entry 2 only needs to be calculate once, since the 
    # MC model doesn't change across different ve sizes.
    for ii in range(nFiles):
        
        # get the current evoSim object. The evoSim object has the mc model, 
        # and output files needed for the figure data.
        evoFile = os.path.join(figSetup['workDir'],dataFiles['sim_snapshot'].iloc[ii]) 
        print("Processing: %s" % (evoFile))
        
        # 1. Returns arrays to calculate fitness changes vs T percent change
        # for the average state
        # - ia: average state 
        # - ba: b-term at average state
        # - vp: ve as percent of va=vc value for mc model
        # - fc: fitness difference attractor thry and average state
        # - bs: b-term at attractor
        # - ds: d-term at attractor
        #
        # Usage: figDataSet[key][datakey][varkey][idx]
        #
        
        # save the data to a pickle file
        with open(evoFile, 'rb') as file:
            # Serialize and write the variable to the file
            evoSim = pickle.load(file)
            
        evoSim.outputStatsFileBase      = evoSim.outputStatsFileBase.replace('sim_Fig4','sim_Fig5').replace('_LoHiRho','')
        evoSim.outputSnapshotFileBase   = evoSim.outputSnapshotFileBase.replace('sim_Fig4','sim_Fig5').replace('_LoHiRho','')
        evoSim.outputStatsFile          = evoSim.outputStatsFile.replace('sim_Fig4','sim_Fig5').replace('_LoHiRho','')
        evoSim.outputSnapshotFile       = evoSim.outputSnapshotFile.replace('sim_Fig4','sim_Fig5').replace('_LoHiRho','')
        evoSim.simDatDir                = evoSim.simDatDir.replace('_LoHiRho','')
        
        evoSim.simInit.outputStatsFileBase      = evoSim.simInit.outputStatsFileBase.replace('sim_Fig4','sim_Fig5').replace('_LoHiRho','')
        evoSim.simInit.outputSnapshotFileBase   = evoSim.simInit.outputSnapshotFileBase.replace('sim_Fig4','sim_Fig5').replace('_LoHiRho','')
        evoSim.simInit.simDatDir                = evoSim.simInit.simDatDir.replace('_LoHiRho','')
        
        # overwrite the existing simfile
        with open(evoSim.get_evoSimFilename(), 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(evoSim, file)
            
    return None

# --------------------------------------------------------------------------

def main():
    # main method to run the various simulations
    
    ###############################################################
    ########### Setup file paths and filename parameters ##########
    ###############################################################
    # filepaths for loading and saving outputs
    # inputsPath  = os.path.join(os.getcwd(),'inputs')
    figSetup = dict()
    figSetup['outputsPath'] = os.path.join(os.getcwd(),'outputs')
    figSetup['figSavePath'] = os.path.join(os.getcwd(),'figures','MainDoc')
    
    # filenames and paths for saving outputs
    figSetup['saveFigFile'] = 'fig_bEvo_DRE_traitInterference_increaseT.pdf'
    figSetup['simDatDir']   = 'sim_bEvo_DRE_TFitChng_NewTest'
    figSetup['workDir']     = os.path.join(figSetup['outputsPath'],figSetup['simDatDir'])
    
    # set the output files to load 
    figSetup['dataList'] = 'simList_bEvo_DRE_fitnessGain_traitInterference_TFitChng.csv'
    
    # set the name of the output file that will store the processed data
    # after processing data an initial time, we check for this file to avoid 
    # reprocessing the data again.
    figSetup['dataFile'] = figSetup['dataList'].replace('.csv', '_saveDat.pickle')
    figSetup['saveData'] = os.path.join(figSetup['workDir'],figSetup['dataFile']) 
    figSetup['panelDef'] = {'A':'LoRho','B':'MeRho','C':'HiRho'}

    ###############################################################
    ########### Run the simulations / load simulation data ########
    ###############################################################
    # get the sim data from the file list, function will carry out the calculations
    # needed for plots and return them as a list for each figset.
    #
    # Note: the first time we process the data, we save it in the outputs 
    #       directory. If the the file exist, then use the save file, otherwise
    #       process the data.
    
    # first read the data list and get the set of fig panels
    dataFiles = pd.read_csv(os.path.join(figSetup['workDir'],figSetup['dataList']))
    panel_set = list(np.unique(dataFiles['fig_panel'].values))
    
    for panelkey in panel_set:
        
        # start timer
        tic = time.time()
        
        # get the date for the figure
        fix_evoFiles(figSetup,panelkey)
            
        print(time.time()-tic)
    
if __name__ == "__main__":
    main()
    

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:08:31 2022
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

# import matplotlib.pyplot as plt
import numpy as np

import time
import pickle 

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')
  
import evoLibraries.SimRoutines.SIM_functions as mysim
from evoLibraries.MarkovChain import MC_factory as mcFac

#%% ------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------

def run_pfix_sim_estimator(idx,mcModel):
    
    bwt         = mcModel.bi[idx] 
    bmt         = mcModel.bi[idx+1]
    cwt         = 1
    cmt         = 1+mcModel.params['cp']
    dwt         = mcModel.params['d']
    
    b_bFix      = np.array([ bwt, bmt ])
    c_bFix      = np.array([ cwt, cwt ])
    d_bFix      = np.array([ dwt, dwt ])
    
    b_cFix      = np.array([ bwt, bwt ])
    c_cFix      = np.array([ cwt, cmt ])
    d_cFix      = np.array([ dwt, dwt ])
    
    init_pop    = [int(mcModel.eq_Ni[idx]-1), 1]
    
    nPfix       = 1
    fixThrshld  = 1e3
    

    # run simulation for pfix estimate b-mutation
    pFixSimEst_b = mysim.estimate_popEvo_pFix(mcModel.params,init_pop,\
                                                   b_bFix, d_bFix, c_bFix, nPfix, fixThrshld)
    
    # run simulation for pfix estimate b-mutation
    pFixSimEst_c = mysim.estimate_popEvo_pFix(mcModel.params,init_pop,\
                                                   b_cFix, d_cFix, c_cFix, nPfix, fixThrshld)
            
    return [pFixSimEst_b, pFixSimEst_c]

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# filepaths for loading and saving outputs
inputsPath  = os.path.join(os.getcwd(),'inputs')
outputsPath = os.path.join(os.getcwd(),'outputs')
figSavePath = os.path.join(os.getcwd(),'figures','Supplement')

# filenames and paths for saving outputs
figFile     = 'fig_bEvo_DRE_pFix_plots.pdf'
figDatDir   = 'fig_bEvo_DRE_pFxPlt_pfix1'
paramFile   = ['evoExp_DRE_bEvo_01_parameters_DEBUG.csv']
paramTag    = ['param_01_DRE_bEvo']
saveDatFile = [''.join(('_'.join((figDatDir,pTag)),'.pickle')) for pTag in paramTag]

# set paths to generate output files for tracking progress of loop/parloop
mcModelOutputPath   = os.path.join(outputsPath,figDatDir) 
figFilePath         = os.path.join(figSavePath,figFile)

#%% ------------------------------------------------------------------------
# generate MC data
# --------------------------------------------------------------------------

# define model list (empty), model type and abs fitness axis.
mcModels    = []
modelType   = 'DRE'
absFitType  = 'bEvo'
nMc         = len(paramFile)

# get the mcArray data
if not (os.path.exists(mcModelOutputPath)):
    # if the data does not exist then generate it
    os.mkdir(mcModelOutputPath)
    
    # start timer
    tic = time.time()
    
    # generate models
    for ii in range(nMc):
        # generate mcModels
        mcModels.append(mcFac.mcFactory().newMcModel( os.path.join(inputsPath,paramFile[ii]), \
                                                   modelType,absFitType))
        # ----------------------
        # Caculating 2 pfix estimates
        # ----------------------
        iCheck = mcModels[-1].get_mc_stable_state_idx()
        pFixSimEst = run_pfix_sim_estimator(iCheck,mcModels[-1])
        
        # save the data to a pickle file
        with open(os.path.join(mcModelOutputPath,saveDatFile[ii]), 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump([mcModels[-1],pFixSimEst], file)
    
    print(time.time()-tic)

else:
    # load mcModel data
    for ii in range(nMc):
        # if data exist, then just load it to generate the figure
        with open(os.path.join(mcModelOutputPath,saveDatFile[ii]), 'rb') as file:
            # Serialize and write the variable to the file
            mcModels.append(pickle.load(file))

    
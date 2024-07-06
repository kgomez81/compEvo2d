# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:09:10 2022

@author: Kevin Gomez

Script to run a full simulation of 2d evolution with relative and absolute fitness

"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle 

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.MarkovChain import MC_array_class as mcArry
import figFunctions as figFun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

# filepaths for loading and saving outputs
inputsPath  = os.path.join(os.getcwd(),'inputs')
outputsPath = os.path.join(os.getcwd(),'outputs')
figSavePath = os.path.join(os.getcwd(),'figures','Supplement')

# filenames for saving outputs
figFile     = 'fig_bEvo_DRE_Rho_vs_ScSa_UaUc_varyUcSb0.pdf'
figDatDir   = 'fig_bEvo_DRE_RhoUcSb0_pfix2'
paramFile   = 'evoExp_DRE_bEvo_09_parameters.csv'
paramTag    = 'param_09_DRE_bEvo'
saveDatFile = ''.join(('_'.join((figDatDir,paramTag)),'.pickle'))

# set paths to generate output files for tracking progress of loop/parloop
mcArrayOutputPath   = os.path.join(outputsPath,figDatDir) 
saveDatFilePath     = os.path.join(mcArrayOutputPath,saveDatFile)
figFilePath         = os.path.join(figSavePath,figFile)

# The parameter file is read, and a dictionary with their values is generated.
paramFilePath = os.path.join(inputsPath,paramFile)
modelType   = 'DRE'
absFitType  = 'bEvo'
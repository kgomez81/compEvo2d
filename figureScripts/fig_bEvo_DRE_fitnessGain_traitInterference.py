# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: Kevin Gomez
Script to generate plot demonstrating the degree of inteference between relative
and absolute fitness traits.
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

# helper functions to manage data and generate plots
import figFunctions as figfun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

def create_traitInterferenceFig(figData):
    # create_traitInterferenceFig() generates the main figure showing the 
    # degree of interference between the relative and absolute fitness traits.
    
    fig,ax = plt.subplots(1,1,figsize=[5,5])
    ax.plot(figData['noInt'][:,0], figData['noInt'][:,1],c='blue',marker='o', label='No Interference')
    ax.plot(figData['perfInt'][:,0], figData['perfInt'][:,1],c='red',marker='o',label='Perfect Stalling')
    ax.plot(figData['EstInt'][:,0], figData['EstInt'][:,1],c='black',marker='o',label='Est. Interference')
    ax.set_xlabel('Territory Size (Log10)')
    ax.set_ylabel('Absolute Fitness State')
    ax.legend()
    
    fig.savefig(figData['figSaveName'],bbox_inches='tight')
    
    return None

def get_simData(evoSimFile):
    # get_simData() collects the simulation data necessary to plot the figure
    
    # load evoSim 
    evoSim = figfun.get_evoSnapshot(evoSimFile)
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # get the average state
    bidx_avg = data['mean_b_idx'].values
    idxav = np.mean(bidx_avg)

    # get the different vb=vc, vb=ve intersections
    idxVbVe = np.max(evoSim.mcModel.get_va_ve_intersection_index())
    idxVbVe = evoSim.mcModel.state_i[int(idxVbVe)]
    
    idxVbVc = np.max(evoSim.mcModel.get_va_vc_intersection_index())
    idxVbVc = evoSim.mcModel.state_i[int(idxVbVc)]
    
    # get the simulation T value
    terrSize = np.log10(evoSim.params['T'])

    # set data as outputs
    simData = []    
    simData.append([terrSize,idxVbVe])  # no interference
    simData.append([terrSize,idxVbVc])  # perfect interference
    simData.append([terrSize,idxav])    # sim est interference
    
    return simData

def main():
    # main method to run the various simulation
    # filepaths for loading and saving outputs
    # inputsPath  = os.path.join(os.getcwd(),'inputs')
    outputsPath = os.path.join(os.getcwd(),'outputs')
    figSavePath = os.path.join(os.getcwd(),'figures','MainDoc')
    
    figData = dict()
    
    # filenames and paths for saving outputs
    saveFigFile = 'fig_bEvo_DRE_fitnessGain_traitInterference.pdf'
    simDatDir   = 'sim_bEvo_DRE_Fig4'
    
    figData['figSaveName'] = os.path.join(figSavePath,saveFigFile)
    
    simDatFiles = []
    simDatFiles.append('sim_Fig2A_1E9_snpsht_param_03_DRE_bEvo_1E9_20250704_114607_evoSim.pickle')
    simDatFiles.append('sim_Fig2A_5E9_snpsht_param_03_DRE_bEvo_5E9_20250704_114750_evoSim.pickle')
    simDatFiles.append('sim_Fig2A_1E10_snpsht_param_03_DRE_bEvo_1E10_20250704_114945_evoSim.pickle')
    simDatFiles.append('sim_Fig2A_5E10_snpsht_param_03_DRE_bEvo_5E10_20250704_115142_evoSim.pickle')
    
    
    noInterference      = []
    perfectInterference = []
    simEstInterference  = []
    
    for ii in range(len(simDatFiles)):
        # set up sim initialization items that vary
        simFile = os.path.join(outputsPath,simDatDir,simDatFiles[ii])
        simData = get_simData(simFile)
        
        # assign the data
        noInterference.append(simData[0])
        perfectInterference.append(simData[1])
        simEstInterference.append(simData[2])

    figData['noInt'] = np.asarray(noInterference)
    figData['perfInt'] = np.asarray(perfectInterference)
    figData['EstInt'] = np.asarray(simEstInterference)
    
    create_traitInterferenceFig(figData)
    
if __name__ == "__main__":
    main()
    
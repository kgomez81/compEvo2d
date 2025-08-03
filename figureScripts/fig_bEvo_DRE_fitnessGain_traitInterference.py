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

from evoLibraries.LotteryModel import LM_functions as lmfun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

def create_traitInterferenceFig(figDataSet):
    # create_traitInterferenceFig() generates the main figure showing the 
    # degree of interference between the relative and absolute fitness traits.
    
    fig,ax = plt.subplots(3,1,figsize=[5,5])
    
    for ii in range(len(figDataSet)):
        
        figData = figDataSet[ii]
    
        ax[ii].plot(figData['ve_perc'], figData['fit_int2env'],c='blue',marker='o', label='No Interference')
        ax[ii].plot(figData['ve_perc'], np.zeros(figData['ve_perc'].shape),c='red',marker='o',label='Perfect Stalling')
        ax[ii].plot(figData['ve_perc'], figData['fit_int2avg'],c='black',marker='o',label='Est. Interference')
        ax[ii].set_ylabel('Fitness Increase')
        ax[ii].set_xlim([10,80])
        ax[ii].set_xticks([10*ii for ii in range(1,9)])
        
        if (ii==len(figDataSet)-1):
            ax[ii].set_xlabel('ve/v*(vb=vc)')
            ax[ii].set_xticklabels([("%d%%" % (10*ii)) for ii in range(1,9)])
            ax[ii].legend()
        else:
            ax[ii].set_xticklabels(['' for ii in range(1,9)])
    
    # fig.savefig(figData['figSaveName'],bbox_inches='tight')
    
    return None

def get_simData(evoSimFile):
    # get_simData() collects mean state of run, along with the two 
    # intersections for va=vc (no ve) and va=ve (no vc), and fitness
    # from intersection
    
    # load evoSim 
    evoSim = figfun.get_evoSnapshot(evoSimFile)
    
    # load stat file with sampled data
    statsFile = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.outputStatsFile)[1] )
    data = pd.read_csv(statsFile)
    
    # get the average state
    bidx_avg = data['mean_b_idx'].values
    idxAvg = np.mean(bidx_avg)

    # get the different vb=vc, vb=ve intersections
    idxVbVe = np.max(evoSim.mcModel.get_va_ve_intersection_index())
    idxVbVe = evoSim.mcModel.state_i[int(idxVbVe)]
    
    idxVbVc = np.max(evoSim.mcModel.get_va_vc_intersection_index())
    idxVbVc = evoSim.mcModel.state_i[int(idxVbVc)]
    
    
    # estimate fitness changes
    biInt = evoSim.mcModel.bi[int(idxVbVc)]
    biEnv = evoSim.mcModel.bi[int(idxVbVe)]
    biAvg = evoSim.mcModel.bi[int(idxAvg)]
    dTerm = evoSim.mcModel.di[int(idxAvg)]
    
    fitChng_Int2Avg     = lmfun.get_b_SelectionCoeff(biInt,biAvg,dTerm)
    fitChng_Int2Env     = lmfun.get_b_SelectionCoeff(biInt,biEnv,dTerm)
    fitChng_Avg2Env     = lmfun.get_b_SelectionCoeff(biAvg,biEnv,dTerm)
    

    # set data as outputs
    simData = []    
    simData.append(idxAvg)      # sim est interference
    simData.append(idxVbVe)     # no interference
    simData.append(idxVbVc)     # perfect interference
    
    simData.append(fitChng_Int2Avg)     # fit change va=vc to avg
    simData.append(fitChng_Int2Env)     # fit change va=vc to va=ve
    simData.append(fitChng_Avg2Env)     # fit change avg to va=ve
    
    return simData


def get_figData(figSetup,ii):
    
    # get list of files
    
    fileList = os.path.join(figSetup['outputsPath'],figSetup['simDatDir'],figSetup['dataList'][ii])
    dataFiles = pd.read_csv(fileList)
    nFiles = len(dataFiles)
    
    # array to collect results in
    data = []
    
    # loop through files and collection the data needed
    for ii in range(nFiles):
        # get the mean and median state of the run, as well as the other two
        # intersection points: va=vc, va=ve, estimate rho
        evoFile = os.path.join(figSetup['outputsPath'],figSetup['simDatDir'],os.path.split(dataFiles['sim_snapshot'][ii])[1]) 
        print(evoFile)
        data.append([float(dataFiles['ve_percent'][ii])] + get_simData(evoFile))
    
    data = np.asarray(data)
    
    # after collecting the data, convert it into useful arrays for plotting
    figData = dict()
    figData['ve_perc'] = data[:,0]
    figData['idx_avg'] = data[:,1]
    figData['idx_int'] = data[:,2]
    figData['idx_env'] = data[:,3]
    figData['fit_int2avg'] = data[:,4]
    figData['fit_int2env'] = data[:,5]
    figData['fit_avg2env'] = data[:,6]
    
    return figData

def main():
    # main method to run the various simulation
    # filepaths for loading and saving outputs
    # inputsPath  = os.path.join(os.getcwd(),'inputs')
    figSetup = dict()
    figSetup['outputsPath'] = os.path.join(os.getcwd(),'outputs')
    figSetup['figSavePath'] = os.path.join(os.getcwd(),'figures','MainDoc')
    
    # filenames and paths for saving outputs
    figSetup['saveFigFile'] = 'fig_bEvo_DRE_fitnessGain_traitInterference.pdf'
    figSetup['simDatDir']   = 'sim_bEvo_DRE_VeFitChng'
    
    figSetup['dataList'] = []
    figSetup['dataList'].append('simList_bEvo_DRE_fitnessGain_traitInterference_lowRho.csv')
    # figSetup['dataList'].append('simList_bEvo_DRE_fitnessGain_traitInterference_medRho.csv')
    # figSetup['dataList'].append('simList_bEvo_DRE_fitnessGain_traitInterference_highRho.csv')
    
    nFig = len(figSetup['dataList'])
    figDatSet = []
    
    for ii in range(nFig):
        # get the sim data from the file list
        figDatSet.append(get_figData(figSetup,ii))
        
    # create the figure
    create_traitInterferenceFig(figDatSet)
    
if __name__ == "__main__":
    main()
    
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

# --------------------------------------------------------------------------

def get_simData(evoSimFile):
    # get_simData() collects mean state of run for a given ve size. The 
    # output is mean state (int) and the fitness increase associated with this
    # state relative to the va=vc theoretical intersections
    # 
    # outputs include:
    # - idxVbVc             # perfect interference
    # - idxAvg              # sim est interference
    # - biInt               # bi at va=vc intersection
    # - biAvg               # bi at average abs state
    # - dTerm               # d term of mc model for reference
    # - fitChng_Int2Avg     # fit change va=vc to avg
    
    
    # get the evoSim object. We do it locally, becuase the evoSim object may
    # not necessarily store the current working directory path, since the sim 
    # runs sometimes occur on different machines and keep those paths with the 
    # snapshot file.
    evoSim = figfun.get_evoSnapshot(evoSimFile)
    
    # load stat file with sampled data
    statsFile = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.outputStatsFile)[1] )
    data = pd.read_csv(statsFile)
    
    # get the average state
    bidx_avg = data['mean_b_idx'].values
    idxAvg = np.mean(bidx_avg)
    
    # get the different vb=vc intersections
    idxVbVc = np.max(evoSim.mcModel.get_va_vc_intersection_index())
    idxVbVc = evoSim.mcModel.state_i[int(idxVbVc)]
    
    # estimate fitness changes
    biInt = evoSim.mcModel.bi[int(idxVbVc)]
    biAvg = evoSim.mcModel.bi[int(idxAvg)]
    dTerm = evoSim.mcModel.di[int(idxAvg)]
    
    fitChng_Int2Avg     = lmfun.get_b_SelectionCoeff(biInt,biAvg,dTerm)

    # set data as outputs
    simData = []    
    simData.append(idxVbVc)             # perfect interference
    simData.append(idxAvg)              # sim est interference
    simData.append(biInt)               # bi at va=vc intersection
    simData.append(biAvg)               # bi at average abs state
    simData.append(dTerm)               # d term of mc model for reference
    simData.append(fitChng_Int2Avg)     # fit change va=vc to avg
    
    return simData

# --------------------------------------------------------------------------

def get_mcModelIntersect(evoSimFile):
    # get_mcModelIntersect() takes an MC model and generates intersections 
    # of ve with va values of the state space.
    
    # load evoSim object to ge tthe mcModel
    evoSim = figfun.get_evoSnapshot(evoSimFile)    
    
    # first get va=vc intersection index
    idxVbVc = np.max(evoSim.mcModel.get_va_vc_intersection_index())
    
    iSInt   = evoSim.mcModel.state_i[idxVbVc]
    veInt   = evoSim.mcModel.va_i[idxVbVc]
    biInt   = evoSim.mcModel.bi[idxVbVc]
    dTerm   = evoSim.mcModel.di[idxVbVc]
    
    # now get all of the va's, bi above the intersection
    ib = evoSim.mcModel.state_i [idxVbVc:]
    bi = evoSim.mcModel.bi      [idxVbVc:]
    va = evoSim.mcModel.va_i    [idxVbVc:]
    
    # clean up the data for only non negative va
    ib = ib[va>0]
    bi = bi[va>0]
    va = va[va>0]
    
    # calculate the fitness increases along the possible va(=ve)
    fitChng = np.zeros(va.shape)
    vePerc  = np.zeros(va.shape)
    
    iInter   = iSInt * np.ones(va.shape)
    bInter   = biInt * np.ones(va.shape)
    vInter   = 100 * np.ones(va.shape)
    fInter   = np.zeros(va.shape)
    
    for ii in range(va.shape[0]):
        fitChng[ii] = lmfun.get_b_SelectionCoeff(biInt,bi[ii],dTerm)
        vePerc[ii]  = va[ii]/veInt
    
    # set data as outputs
    mcIntData = []    
    mcIntData.append(ib)        # sim states associated with selected ve
    mcIntData.append(bi)        # bi values associated with selected ve
    mcIntData.append(va)        # actual values of ve
    mcIntData.append(vePerc)    # ve as percent of va=vc thry value 
    mcIntData.append(fitChng)   # fitness changes with shift to intersection
    
    # note: no fitness changes with perfect interference but these are being 
    # added for convenience to generate the figure
    mcIntData.append(iInter)    # intersection state
    mcIntData.append(bInter)    # bi value at intersection
    mcIntData.append(vInter)    # v value at intersection 
    mcIntData.append(fInter)    # zeros because intersection doesn't change with ve

    myKeys = ['ib','bi','vePerc','fitChng','i_star','b_star','v_star','f_star']
    
    simData = dict(zip(myKeys,mcIntData))
    
    return simData

# --------------------------------------------------------------------------


def get_effectiveVaVc(evoSimFile):
    # get_effectiveVaVc() calculates estimates of va and vc from the adaptive 
    # event log files.
    
    # load evoSim object to ge tthe mcModel
    evoSim = figfun.get_evoSnapshot(evoSimFile)   
    
    # load abs log file and calculate the va estiamtes
    absLogFile  = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.get_adaptiveEventsLogFilename('abs'))[1] )
    dataAbs     = pd.read_csv(absLogFile)
    vaSimEst    = figfun.get_estimateRateOfAdaptFromSim(dataAbs,'abs',evoSim.mcModel)
    
    # load rel log file and calculate the vc estimates
    relLogFile  = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.get_adaptiveEventsLogFilename('rel'))[1] )
    dataRel     = pd.read_csv(relLogFile)
    vcSimEst    = figfun.get_estimateRateOfAdaptFromSim(dataRel,'rel',evoSim.mcModel)
    
    # save the estimates to a dictionary for plotting
    simVaVcEst = dict({'vaEst': vaSimEst, 'vcEst': vcSimEst})
    
    return simVaVcEst

# --------------------------------------------------------------------------

def get_figData(figSetup):
    # get_figData() loops throught the various panel data files and calculates
    # the required figure data. It requires a dictionary with the output
    # directory and filenames where data is store for a simulation run.
    
    # get list of files
    workdir = os.path.join(figSetup['outputsPath'],figSetup['simDatDir'])
    fileList = os.path.join(workdir,figSetup['dataList'])
    dataFiles = pd.read_csv(fileList)
    
    # this will return the number of seperate runs associated with 
    # a particular sim run set
    nFiles = len(dataFiles)
    
    # this will get a list of the panels for reference, eg. ['A','B','C']
    # the panels should be generated in order by the sim execution file.
    panel_set = np.unique(dataFiles['fig_panel'].values)
    
    # Create an array to collect simulation results in. The data will be stored
    # as a dictionary consisting of [keys = Panels, [DataSet1,DataSet2]]
    #
    # 1. Panel will just be the char for the panel
    # 2. DataSet will be a dictionary consisting of a couple of things:
    #
    #     - 1st Entry is array of va=ve (estimate) vs sa (relative to va=vc point)
    #       Calculated from get_simData()
    #
    #     - 2nd Entry is array of max/min sa pairs from mc ve-va/vc intersections
    #       Calculated from get_mcModelIntersect()
    #
    #     - 3rd Entry is array of effective va/vc values for given ve size
    #       Calculated from get_mcModelIntersect()
    #
    figData = dict.fromkeys(panel_set)
    for key in figData.keys():
        figData[key] = [[],[],[]]
        
    # Now we loop through each file, get the data sets for entry 1 and entry 2.
    #
    # However, we note that entry 2 only needs to be calculate once, since the 
    # MC model doesn't change across different ve sizes.
    #
    for ii in range(nFiles):
        
        # get the current evoSim object. The evoSim object has the mc model, 
        # and output files needed for the figure data.
        evoFile = os.path.join(workdir,dataFiles['sim_snapshot'][ii]) 
        print("Processing: %s" % (evoFile))
        
        # get the current panel
        crntKey = dataFiles['fig_panel'][ii]
        
        # calculate the 1st entry of data set for current file and append
        # to respective dictionary entry for panel. This will be turned into 
        # an array of data once all of the files have been processed.
        figData[crntKey][0].append(get_simData(evoFile))
    
        # check if the mc model still hasn't been populated for this panel
        if (figData[crntKey][1] == []):
            figData[crntKey][1].append(get_mcModelIntersect(evoFile))
            
        # calculate the third entry for va vc estimates
        figData[crntKey][2].append(get_effectiveVaVc(evoFile))
    
    return figData

# --------------------------------------------------------------------------

def main():
    # main method to run the various simulation
    # filepaths for loading and saving outputs
    # inputsPath  = os.path.join(os.getcwd(),'inputs')
    figSetup = dict()
    figSetup['outputsPath'] = os.path.join(os.getcwd(),'outputs')
    figSetup['figSavePath'] = os.path.join(os.getcwd(),'figures','MainDoc')
    
    # filenames and paths for saving outputs
    figSetup['saveFigFile'] = 'fig_bEvo_DRE_fitnessGain_traitInterference_veFitChng.pdf'
    figSetup['simDatDir']   = 'sim_bEvo_DRE_VeFitChng'
    
    # set the output files to load 
    figSetup['dataList'] = 'simList_bEvo_DRE_fitnessGain_traitInterference_veFitChng.csv'
    
    # get the sim data from the file list, function will carry out the calculations
    # needed for plots and return them as a list for each figset
    figDatSet=get_figData(figSetup)
        
    # create the figure
    create_traitInterferenceFig(figDatSet)
    
if __name__ == "__main__":
    main()
    
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:30:20 2022

@author: Kevin Gomez
Script to generate plot demonstrating the degree of inteference between relative
and absolute fitness traits. This script will generate a plot of the fitness 
gains associated different degrees of interference. The plots are generated 
from simulation data captured in the folder 'sim_bEvo_DRE_VeFitChng_NewTest'.
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import time
import pickle 

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

# helper functions to manage data and generate plots
import figFunctions as figfun

from evoLibraries.LotteryModel import LM_functions as lmfun

#%% ------------------------------------------------------------------------
# Get parameters/options
# --------------------------------------------------------------------------

def create_traitInterferenceFig(figDataSet,figSaveName):
    # create_traitInterferenceFig() generates the main figure showing the 
    # degree of interference between the relative and absolute fitness traits.
    # 
    # Description of figDataSet
    # - figDataSet is a dictionary with panel names as keys (e.g 'A','B',...)
    #   each panel entry is a list of three items:
    #
    #   1. Dictionary with va average's across different ve levels.
    #   2. Dictionary with MC state data for ve=va (no interf) data 
    #   3. Dictionary with average va and vc estimates
    
    panels = list(figDataSet.keys())
    nPan = len(panels)
    
    fig,ax = plt.subplots(nPan,1,figsize=[5,nPan*3])
    
    for ii, panel in enumerate(panels):
        
        figData = process_sim_data_for_plots(figDataSet[panel])
        myYlims = get_ylims(figData)
        
        # plot the fitness increase of the average va for each ve level
        ax[ii].scatter(figData['bAvg']['ve_perc'], figData['bAvg']['fit_chng'],c='black',marker='o',label='Imperfect Stalling')
        
        # plot the fitness increase of the intersection va=vc curve
        ax[ii].plot(figData['bEnv']['ve_perc'], figData['bEnv']['fit_chng'],c='blue' ,label='No Interference')
        
        # plot the fitness increase of the intersection va=vc curve
        ax[ii].plot(figData['bInt']['ve_perc'], figData['bInt']['fit_chng'],c='red' ,label='Perfect Interference')
        
        ax[ii].set_ylabel('Fitness Increase')
        ax[ii].set_ylim(myYlims)
        ax[ii].set_xlim([0,100])
        ax[ii].set_xticks([10*ii for ii in range(0,11)])
        
        ax[ii].text(3,0.9*myYlims[1],("(%s)"%(panel)),fontsize = 12) 
        ax[ii].text(15,0.9*myYlims[1],(r'$\rho$=%.2f'%(figData['rho'])),fontsize = 12) 
        
        if (ii==len(figDataSet)-1):
            ax[ii].set_xlabel(r'$v_E/v_a*$')
            ax[ii].set_xticklabels([("%d%%" % (10*ii)) for ii in range(0,11)])
            
        elif (ii==0):
            ax[ii].legend()
            ax[ii].set_xticklabels(['' for ii in range(0,11)])
        else:
            ax[ii].set_xticklabels(['' for ii in range(0,11)])
    plt.tight_layout()
    fig.savefig(figSaveName,bbox_inches='tight')
    
    return None

# --------------------------------------------------------------------------

def create_singleTraitInterferenceFig(figDataSet,figSaveName):
    # create_singleTraitInterferenceFig() generates the main figure showing the 
    # degree of interference between the relative and absolute fitness traits.
    # 
    
    panels = list(figDataSet.keys())
    
    fig,ax = plt.subplots(1,1,figsize=[7,7])
        
    figData = process_sim_data_for_plots(figDataSet['B'])
    myYlims = get_ylims(figData)
    
    # plot the fitness increase of the average va for each ve level
    ax.scatter(figData['bAvg']['ve_perc'], figData['bAvg']['fit_chng'],c='black',marker='o',label='Imperfect Stalling')
    ax.plot(figData['bEnv']['ve_perc'], figData['bEnv']['fit_chng'],c='blue',linestyle='-.',label='No Interference')
    ax.plot(figData['bInt']['ve_perc'], figData['bInt']['fit_chng'],c='red' ,linestyle='--',label='Intersection')
    
    ax.set_ylabel('Change in Fitness (w.r.t Intersection)',fontsize=20,labelpad=8)
    ax.set_ylim(myYlims)
    ax.set_xlim([0,100])
    
    ax.set_xlabel(r'$v_E/v_a^*$',fontsize=20,labelpad=8)
    yticks = list(ax.get_yticks())
    yticks = [round(elmnt,1) for elmnt in yticks]
    yticklabels = list(map(str,yticks))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=14)
    
    ax.set_xticks([20*ii for ii in range(0,6)])
    ax.set_xticklabels([("%d%%" % (20*ii)) for ii in range(0,6)],fontsize=14)
    
    ax.legend(fontsize=20)
        
    plt.tight_layout()
    
    figSaveName = figSaveName.replace('.pdf','_rho1.pdf')
    fig.savefig(figSaveName,bbox_inches='tight')
    
    return None

# --------------------------------------------------------------------------

def process_sim_data_for_plots(figDataSet):
    # function use to extract the specific sim data needed for the plot. Here
    # we organize the data retreived from get_figData. This is the data needed
    #
    # 1. ve percent vs fitness change from attractor to va average 
    #
    # 2. ve percent vs fitness change from attractor to ve
    #    
    
    figData = {'bAvg':[],'bEnv':[],'bInt':[],'rho':[]}
    
    # Part 1 - Uses dict-1 of figDataSet, Sim average va with fitness increase
    #          vars: 'i_star','i_avg','b_star','b_avg','d_term','fit_chng'
    figData['bAvg'] = {'ve_perc': figDataSet[0]['ve_perc'], 'fit_chng': figDataSet[0]['fit_chng']}
    
    # Part 2 - Uses dict-2 of figDataSet, va=ve fitness increase data
    #          vars: 'ib','bi','ve','ve_perc','fit_chng'
    figData['bEnv'] = {'ve_perc': figDataSet[1]['ve_perc'], 'fit_chng': figDataSet[1]['fit_chng']}
    
    # # Part 3 - Perfect stalling
    ve_perc_pstall  = figDataSet[1]['ve_perc']
    fig_chng_pstall = np.zeros(ve_perc_pstall.shape)
    figData['bInt'] = {'ve_perc': ve_perc_pstall, 'fit_chng': fig_chng_pstall}
    
    figData['rho']  = figDataSet[1]['rho']
    
    return figData

# --------------------------------------------------------------------------

def get_ylims(figData):
    # simple function to get the ylim values
    
    ymax = np.max(figData['bEnv']['fit_chng'])
    ymin = np.min(figData['bAvg']['fit_chng'])
    
    ymax = np.ceil(ymax*10.0)/10.0
    ymin = np.sign(ymin)*np.ceil(np.abs(ymin)*10.0)/10.0
    
    return [ymin,ymax]

# --------------------------------------------------------------------------

def get_simData(evoSimFile,figData_crntPanel):
    # get_simData() collects mean state of run for a given ve size. The 
    # output is mean state (int) and the fitness increase associated with this
    # state relative to the va=vc theoretical intersections
    # 
    # Outputs:
    # i_avg     - idx at average bi
    # b_avg     - average bi
    # v_avg     - va at average bi
    # ve_perc   - ve percent of va=vc value
    # fit_chng  - fitness change from va=vc to average bi
    
    # Get the evoSim object
    evoSim = figfun.get_evoSnapshot(evoSimFile)
    
    # load stat file with sampled data
    statsFile = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.outputStatsFile)[1] )
    data = pd.read_csv(statsFile)
    
    # get the average state
    i_avg = int(np.mean(data['mean_b_idx'].values))
    
    # get intersection and ve values
    i_star = int(np.max(evoSim.mcModel.get_va_vc_intersection_index()))
    v_star = evoSim.mcModel.va_i[i_star]
    ve     =  evoSim.mcModel.ve_i[i_star]
    
    # estimate fitness changes
    b_star      = evoSim.mcModel.bi[int(i_star)]
    b_avg       = evoSim.mcModel.bi[int(i_avg)]
    v_avg       = evoSim.mcModel.va_i[int(i_avg)]
    ve_perc     = ve/v_star*100.0
    dTerm       = evoSim.mcModel.di[int(i_star)]    
    fit_chng    = lmfun.get_b_SelectionCoeff(b_star,b_avg,dTerm)

    # group the new data
    mykeys      = ['i_avg','b_avg','v_avg','ve_perc','fit_chng']
    new_data    = [[i_avg],[b_avg],[v_avg],[ve_perc],[fit_chng]]
    
    if (figData_crntPanel==[]):
        updateFigData = dict(zip(mykeys,new_data))
    else:
        for ikey, key in enumerate(mykeys):
            figData_crntPanel[key].extend(new_data[ikey])
        updateFigData = figData_crntPanel
    
    return updateFigData

# --------------------------------------------------------------------------

def get_mcModelIntersect(evoSimFile):
    # get_mcModelIntersect() takes an MC model and generates intersections 
    # of ve with va values of the state space.
    #
    # Outputs:
    # ib        - idx at va=ve
    # bi        - bi at va=vc
    # ve        - ve at ve=va
    # ve_perc   - ve as percent of va=vc  
    # fit_chng  - fitness increase from va=vc to va=ve
    
    # load evoSim object to ge the mcModel
    evoSim = figfun.get_evoSnapshot(evoSimFile)    
    
    # get va=vc and va=vb intersection indices
    idxVbVc = int(np.max(evoSim.mcModel.get_va_vc_intersection_index()))
    rho     = evoSim.mcModel.calculate_evoRho()
    
    # Store some reference values for calculations
    b_star  = evoSim.mcModel.bi[idxVbVc]
    v_star  = evoSim.mcModel.va_i[idxVbVc]
    dTerm   = evoSim.mcModel.di[idxVbVc]
    
    # now get all of the va's, bi above the intersection
    ib = evoSim.mcModel.state_i [idxVbVc:]
    bi = evoSim.mcModel.bi      [idxVbVc:]
    ve = evoSim.mcModel.va_i    [idxVbVc:]  # value of ve when ve=va
    
    # clean up the data for only non negative va
    ib = ib[ve>0]
    bi = bi[ve>0]
    ve = ve[ve>0]
    
    # calculate the fitness increases along the possible va(=ve)
    fit_chng = np.zeros(ve.shape)
    ve_perc  = np.zeros(ve.shape)
    
    for ii in range(bi.shape[0]):
        ve_perc[ii]  = ve[ii]/v_star*100.0
        fit_chng[ii] = lmfun.get_b_SelectionCoeff(b_star,bi[ii],dTerm)
    
    # set data as outputs
    myKeys      = ['ib','bi','ve','ve_perc','fit_chng','rho']
    mcIntData   = [ ib , bi , ve , ve_perc , fit_chng , rho ]    
    mcModelData = dict(zip(myKeys,mcIntData))
    
    return mcModelData

# --------------------------------------------------------------------------

def get_figData(figSetup):
    # get_figData() loops throught the various panel data files and calculates
    # the required figure data. It requires a dictionary with the output
    # directory and filenames where data is store for a simulation run.
    
    # get list of files which are saved to the csv with all of the sim runs
    # for a group of simulations runs of this figure set.
    fileList = os.path.join(figSetup['workDir'],figSetup['dataList'])
    dataFiles = pd.read_csv(fileList)
    
    # this will return the number of seperate runs associated with 
    # a particular sim run set
    nFiles = len(dataFiles)
    
    # this will get a list of the panels for reference, eg. ['A','B','C']
    # the panels should be generated in order by the sim execution file.
    panel_set = np.unique(dataFiles['fig_panel'].values)
    
    # Create an array to collect simulation results in. The data will be stored
    # as a dictionary consisting of [keys = Panels, [DataSet1,DataSet2,DataSet3]]
    #
    # 1. keys will be the char for the panel
    # 2. DataSet will be a dictionary consisting of a couple of things:
    #
    #     - 1st Entry is array of va=ve (estimate) vs sa (relative to va=vc point)
    #       Calculated from get_simData()
    #
    #     - 2nd Entry is dictionary with MC model data specifying ve-va(vc) 
    #       intersections, and fitness changes associated with decreasing ve.
    #       Calculated from get_mcModelIntersect()
    
    figData = dict.fromkeys(panel_set)
    for key in figData.keys():
        figData[key] = [[],[]]
        
    # Now we loop through each file, get the data sets for entry 1 and entry 2.
    # However, we note that entry 2 only needs to be calculate once, since the 
    # MC model doesn't change across different ve sizes.
    #
    for ii in range(nFiles):
        
        # get the current evoSim object. The evoSim object has the mc model, 
        # and output files needed for the figure data.
        evoFile = os.path.join(figSetup['workDir'],dataFiles['sim_snapshot'][ii]) 
        print("Processing: %s" % (evoFile))
        
        # get the current panel
        crntKey = dataFiles['fig_panel'][ii]
        
        # calculate the 1st entry of data set for current file and append
        # to respective dictionary entry for panel. 
        figData[crntKey][0] = get_simData(evoFile,figData[crntKey][0])
    
        # check if the mc model still hasn't been populated for this panel
        # if it has, then skip appending the data
        if (figData[crntKey][1] == []):
            figData[crntKey][1] = get_mcModelIntersect(evoFile)
            
    
    return figData

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
    figSetup['saveFigFile'] = 'fig_bEvo_DRE_fitnessGain_traitInterference_veFitChng.pdf'
    figSetup['simDatDir']   = 'sim_bEvo_DRE_VeFitChng_NewTest'
    figSetup['workDir']     = os.path.join(figSetup['outputsPath'],figSetup['simDatDir'])
    
    # set the output files to load 
    figSetup['dataList'] = 'simList_bEvo_DRE_fitnessGain_traitInterference_veFitChng.csv'
    
    # set the name of the output file that will store the processed data
    # after processing data an initial time, we check for this file to avoid 
    # reprocessing the data again.
    figSetup['dataFile'] = figSetup['dataList'].replace('.csv', '_saveDat.pickle')
    figSetup['saveData'] = os.path.join(figSetup['workDir'],figSetup['dataFile']) 
    
    
    ###############################################################
    ########### Run the simulations / load simulation data ########
    ###############################################################
    # get the sim data from the file list, function will carry out the calculations
    # needed for plots and return them as a list for each figset.
    #
    # Note: the first time we process the data, we save it in the outputs 
    #       directory. If the the file exist, then use the save file, otherwise
    #       process the data.
    if not (os.path.exists(figSetup['saveData'])):
        # start timer
        tic = time.time()
        
        # get the date for the figure
        figDatSet = get_figData(figSetup)
            
        # save the data to a pickle file
        with open(figSetup['saveData'], 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(figDatSet, file)
            
        print(time.time()-tic)

    else:
        # load mcModel data
        with open(figSetup['saveData'], 'rb') as file:
            # Serialize and write the variable to the file
            figDatSet = pickle.load(file)
            
    # create the figure
    # create_traitInterferenceFig(figDatSet,os.path.join(figSetup['workDir'],figSetup['saveFigFile']))
    
    # create one figure with just the rho=1 case
    create_singleTraitInterferenceFig(figDatSet,os.path.join(figSetup['workDir'],figSetup['saveFigFile']))
    
if __name__ == "__main__":
    main()
    
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def create_fitnessGainVsTincrFig(figDataSet,figSaveName,xAxisType):
    """
    Generates plot showing fitness increase vs increases to T
    Inputs:
        - figDataSet, contains all simulation and mc model data
        - figSaveName
        - xAxisType, string indicating vE percent or log(T/T0) x-axis
    Outputs:
        - figure showing fitness increases/decreases vs percent change in T
    """ 

    # get list of keys and setup things for a color map
    vp0     = list(figDataSet.keys())
    nv      = len(vp0)
    
    # collect the data
    figData = process_sim_data_for_plots(figDataSet,0)
    
    # setup the figure and color map
    fig,ax  = plt.subplots(1,3,figsize=[14,5])   
    
    # set common y-lims and ticks
    Ylim = [-0.005,0.185]
    Ytic = [0.02*ii for ii in range(0,10)]
    Ylblv= [str(round(yval*10,1)) for yval in Ytic]
    Ylble = ['' for ii in range(10)]
    
    # loop through keys and add each subplot
    for idx, key in enumerate(vp0):
        
        idxf = 2-idx # switch the order
        
        if xAxisType == 'vaxis':
            
            # xtick settings (change this if sim parameters change)
            xticDat = {'vals':{}, 'lbls': {}}
            xticDat['vals'][0] = [35+5*ii for ii in range(4)]
            xticDat['vals'][1] = [55+5*ii for ii in range(5)]
            xticDat['vals'][2] = [70+10*ii for ii in range(4)]
            for ii in range(3):
                xticDat['lbls'][ii] = ["%s%%"%(val) for val in xticDat['vals'][ii]]
                
            # plot the fitness increase of the average va for each ve level
            ax[idxf].scatter(figData[key]['ve_perc_init'], figData[key]['fit_chng_avg'],color='black',marker='o',label='Imperfect Interference')
            ax[idxf].scatter(figData[key]['ve_perc_init'], figData[key]['fit_chng_env'],color='blue',marker='o',facecolors='none')
            ax[idxf].scatter(figData[key]['ve_perc_init'], figData[key]['fit_chng_int'],color='red',marker='o',facecolors='none',label='Perfect Interference')
            ax[idxf].plot(figData[key]['ve_perc_crv'], figData[key]['fit_chng_crv'],c='blue',linestyle='-',label='No Interference')
        
        else:
            
            xdata = [1,5,100]
            xticDat = {'vals':{}, 'lbls': {}}
            for ii in range(3):
                xticDat['vals'][ii] = [np.log10(val) for val in xdata]
                xticDat['lbls'][ii] = ["%dT"%(val) for val in xdata]
            
            # plot the fitness increase of the average va for each T level
            ax[idxf].scatter(figData[key]['T_perc_chng'], figData[key]['fit_chng_avg'],color='black',marker='o',label='Imperfect Interference')
            # ax[idxf].scatter(figData[key]['T_perc_chng'], figData[key]['fit_chng_env'],color='blue',marker='o',facecolors='none')
            ax[idxf].scatter(figData[key]['T_perc_chng'], figData[key]['fit_chng_int'],color='red',marker='o',facecolors='none',label='Perfect Interference')
            ax[idxf].plot(figData[key]['T_perc_crv'], figData[key]['fit_chng_crv'],c='blue',linestyle='-',marker='.',label='No Interference')
            # ax[idxf].scatter(figData[key]['T_perc_crv'], figData[key]['fit_chng_crv'],c='blue',marker='+',label='No Interference')
        
        ax[idxf].set_ylim(Ylim)    
        ax[idxf].set_yticks(Ytic)
        if (idxf==0):
            ax[idxf].set_yticklabels(Ylblv,fontsize=14)
        else:
            ax[idxf].set_yticklabels(Ylble,fontsize=14)
        
        
        ax[idxf].set_xticks(xticDat['vals'][idx])
        ax[idxf].set_xticklabels(xticDat['lbls'][idx],fontsize=14)
        xmin = np.min(ax[idxf].get_xlim())
        xmax = np.max(ax[idxf].get_xlim())
        
        if (idxf==2):
            ax[idxf].legend(fontsize=16)
            
        ax[idxf].text(xmin+0.01*(xmax-xmin),0.175,"(%s)"%(chr(65+idxf)),fontsize=16)

    # ax.text(1,-0.16,r'$T=T_0 \times 1,5,100$',fontsize=20)    
    
    if xAxisType == 'vaxis':
        fig.supxlabel(r'$v_E/v_a^*$',y=0.05,fontsize=16)
    else:
        fig.supxlabel(r'Multiples of Reference T ($log_{10}$)',y=0.05,fontsize=16)
        
    fig.supylabel('Change in Fitness',x=0.01,fontsize=16)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.tight_layout()
    
    figSaveName = figSaveName.replace('.pdf','_rho1.pdf')
    fig.savefig(figSaveName,bbox_inches='tight')
    
    return None

# --------------------------------------------------------------------------

def sort_sim_runs(sim_run_pts):
    # used to sort runs based on T-percent values
    # 
    # inputs:
    # - sim_run_pts = list of T-percent values as strings
    
    # get an integer list of T-percent values
    T_perc_int = list(map(int,sim_run_pts))
    
    # sort the T-percent list and get the indices
    idxs = list(np.argsort(T_perc_int))
    
    return idxs

# --------------------------------------------------------------------------

def process_sim_data_for_plots(figDataSet,fig_type):
    """
    Extracts and organizes sim and mc model data for the plots of fig_type. 
    
    Inputs:
        - figDataSet, struct with all processed sim and mc model data
        - fig_type, indicator for data selection
    Outputs:
        - figData, struct that can be used to generate plot of fig_type 
    """
    
    
    # OVERVIEW of figDataSet
    # -----------------------------------------------------------------------
    # 0. Save the current value for precent increase in T
    #    Note: this percentage is the starting value of ve/(va=vc)
    #
    # Usage: figDataSet[key][datakey][idx]
    #
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # 2. Returns fine grid arrays to calculate fitness changes vs T percent 
    # change for the shift in va=ve intersection. This is just the MC model
    # data with key info extracted.intersection
    # - ib: va=ve state
    # - bi: b-term of va=ve state
    # - ve: vE as percentage of va=vc
    # - fc: fitness difference from va=vc to va=ve
    # - ro: rho of mc model
    # - bs: b-term of attractor, va=vc
    # - vs: va at attractor
    # - ds: d-term at attractor (same at va=ve if d constant) 
    #
    # Usage: figDataSet[key][datakey][varkey][idx]
    #
    # -----------------------------------------------------------------------
    # 3. Returns arrays to plot curve of fitness gains vs territory size increases
    # - ib: state for va=ve intersection for curve defined by varying T
    # - bi: b-term for va=ve intersection for curve defined by varying T
    # - vp: vE as a percentage of va=vc intersection. Note latter changes
    #       with adjustments to T
    # - fc: fitness change caused from shifting the va=ve intersection after
    #
    # Usage: figDataSet[key][datakey][varkey]
    #
    # -----------------------------------------------------------------------
    # 4. get corresponding mc model
    # - includes all mc model parameters 
    #
    # Usage: figDataSet[key][datakey][idx]
    #
    # -----------------------------------------------------------------------
    # 5. get VaVcEstimates
    # - list with two elements: [states, vaEst], [states, vcEst]
    #
    # Usage: figDataSet[key][datakey][idx][varky]
    # varky: ['vaEst', 'vcEst']
    #
    # -----------------------------------------------------------------------
    # 6. get sojourn times in states
    # - states, with weights for sojourn times
    #
    # Usage: figDataSet[key][datakey][idx][varid]
    # varid: [idxGrp,idxCnt]
    #
    
    # get list of plot types, determined by starting ve/v*, for default T0
    plot_types = list(figDataSet.keys())
    
    if fig_type == 0:
        # CASE: fitness increase plots
        
        # We organize figData by the plot types, i.e.init vE = 50%, 75%, ...
        figData = dict.fromkeys(plot_types)
        
        # setup data keys for each plot set
        dataKeys = ['ve_perc_init','T_perc_chng','fit_chng_avg','fit_chng_env','fit_chng_int','ve_perc_crv','fit_chng_crv']
        
        # -------------------
        # -- Process Data ---
        # -------------------
        # for this section we need the following entries from figDataset
        # - 0. To group data by vE/v* initial percentages 
        # - 1. To calculate points for change in fitness due to changes in T 
        # - 2. To calculate points ve=va fitness increases from T
        # - 3. To calculate fine grid points for ve=va fitness increase from T
        
        # loop through each key 
        for key in plot_types:
            
            # setup empty list to build up data sets, we start with the initial
            # set of values
            vp0 = np.max(figDataSet[key]['sim_avg']['vp'])
            
            vpi = [ vp0 ]    # init vE percent
            tpc = [   0 ]    # Territory size percent
            fca = [   0 ]    # fitness change due to change in average state
            fce = [   0 ]    # fitness change due to change in ve=va location
            fci = [   0 ]    # fitness change due to change in va=vc location
            
            # get the instance of data for the current plot_type = "fig-key"
            figData[key] = dict.fromkeys(dataKeys)
            
            # We first need to sort the data sets in case the data was read in 
            # a non-increasing order (T-percent). We do this by sorting the 
            # first entry of figDataSet. T_perc = 100% (i.e. no change) should
            # the first idx
            figIdx = sort_sim_runs(figDataSet[key]['T_perc'])
            
            # Get reference b-term for starting T-value (1st index). This will
            # be the b-value at the average state, with no change in T (=100%).
            # We get this data from [1] data of the key. We also get other 
            # values like the b-term for the intersections (va=vc, va=ve)
            id0 = figIdx[0]
            
            di0 = figDataSet[key]['sim_avg']['ds'][id0]  # d-term (doesn't change)
            ba0 = figDataSet[key]['sim_avg']['ba'][id0]  # b-term at average
            
            be0 = figDataSet[key]['mc_int_pts']['bi'][id0]  # b-term at va=ve
            bs0 = figDataSet[key]['mc_int_pts']['bs'][id0]  # b-term at va=vc
            
            for idx in figIdx[1:]:
                
                # Part 1 - get the current b-erms
                ba1 = figDataSet[key]['sim_avg']['ba'][idx]
                be1 = figDataSet[key]['mc_int_pts']['bi'][idx]
                bs1 = figDataSet[key]['mc_int_pts']['bs'][idx]
                
                vpi.append(figDataSet[key]['sim_avg']['vp'][idx])
                tpc.append(np.log10(figDataSet[key]['T_perc'][idx]/100))
                fca.append(lmfun.get_b_SelectionCoeff(ba0,ba1,di0))
                fce.append(lmfun.get_b_SelectionCoeff(be0,be1,di0))
                fci.append(lmfun.get_b_SelectionCoeff(bs0,bs1,di0))
            
            figData[key]['ve_perc_init'] = vpi
            figData[key]['T_perc_chng' ] = tpc
            figData[key]['fit_chng_avg'] = fca
            figData[key]['fit_chng_env'] = fce
            figData[key]['fit_chng_int'] = fci
            
            # Part 2 - get the fine grid array (crv=curves)
            figData[key]['ve_perc_crv' ] = 100*np.asarray(figDataSet[key]['mc_curves']['vp'])
            figData[key]['T_perc_crv'  ] = np.log10(np.asarray(figDataSet[key]['mc_curves']['tp']))
            figData[key]['fit_chng_crv'] = np.asarray(figDataSet[key]['mc_curves']['fc'])
            
    else:
        # CASE: mc plots for the sim runs
        
        # -------------------
        # -- Process Data ---
        # -------------------
        # for this section we need the following entries from figDataset
        # - 0. To group data by vE/v* initial percentages 
        # - 1. To calculate points for change in fitness due to changes in T 
        # - 2. To calculate points ve=va fitness increases from T
        # - 3. To calculate fine grid points for ve=va fitness increase from T
        print('not working yet')
    
    return figData

# --------------------------------------------------------------------------

def process_sim_mc_plots(figDataSet):
    # function use to extract the specific sim data needed for the plot. Here
    # we organize the data retreived from get_figData. This is the data needed
    
    figData = {'mcMod': {'abs':[],'rel':[],'env':[]},'vEst':{'abs':[],'rel':[]},'mcHist':[],'params':[]}
    
    # mc Model values
    crntMcModel = figDataSet[4]
    figData['mcMod']['abs'] = {'ib': crntMcModel.state_i, 'v': crntMcModel.va_i}
    figData['mcMod']['rel'] = {'ib': crntMcModel.state_i, 'v': crntMcModel.vc_i}
    figData['mcMod']['env'] = {'ib': crntMcModel.state_i, 'v': crntMcModel.ve_i}
    
    # v estimates
    figData['vEst']['abs'] = {'ib': figDataSet[5]['vaEst']['ix'], 'v': figDataSet[5]['vaEst']['vx']}
    figData['vEst']['rel'] = {'ib': figDataSet[5]['vcEst']['ix'], 'v': figDataSet[5]['vcEst']['vx']}
    
    # histogram data
    figData['mcHist'] = figDataSet[6]
    
    # mc Model parameters
    figData['params'] = {'ro': crntMcModel.calculate_evoRho(),'vp': figDataSet[1]}
    
    return figData

# --------------------------------------------------------------------------

def get_ylims(figData):
    # simple function to get the ylim values
    
    keys = list(figData.keys())
    ymax = 0
    ymin = 0
    
    for key in keys:
        ymax = np.max([ymax, np.max(figData[key]['fit_chng'])])
        ymin = np.min([ymin, np.min(figData[key]['fit_chng'])])
    
    ymax = np.ceil(ymax*10.0)/10.0
    ymin = np.sign(ymin)*np.ceil(np.abs(ymin)*10.0)/10.0
    
    return [ymin,ymax]

# --------------------------------------------------------------------------

def get_T_sampling(T_perc_arry,nSample,Tscale=None):
    # we use a log scale to determine sampling of T percentages
    
    TpercMin = np.min(T_perc_arry)
    TpercMax = np.max(T_perc_arry)
    
    if (Tscale==None) or (Tscale=='lin'):
        pDelta = (TpercMax - TpercMin)/(nSample-1)
        TvalsPerc = [(TpercMin+ii*pDelta) for ii in range(nSample)]
    else:
        pDelta = np.log10(TpercMax/TpercMin)/(nSample-1)
        TvalsPerc = [TpercMin*10**(ii*pDelta) for ii in range(nSample)]
    
    return TvalsPerc

# --------------------------------------------------------------------------

def get_figData(figSetup):
    # get_figData() loops throught the various panel data files and calculates
    # the required figure data. It requires a dictionary with the output
    # directory and filenames where data is store for a simulation run.
    
    # get list of files which are saved to the csv with all of the sim runs
    # for a group of simulations runs of this figure set.
    fileList = os.path.join(figSetup['workDir'],figSetup['dataList'])
    dataFiles = pd.read_csv(fileList)
    
    # Number of runs in current sim run set, iterating percentage changes in T,
    # with different starting percentages of vE.
    nFiles = len(dataFiles)
    
    # List of panels that correspond to different vE starting percentages
    vperc = np.unique(dataFiles['ve_percent'].values)
    
    # Sampling T-percentage changes for fitness gains without interference. 
    # only 1-set needed for this run set.
    TvalsPerc = get_T_sampling(dataFiles['T_percent'].values,10,'log')
    
    # Create an array to collect simulation results in. 
    #   dictionary of [keys = Panels, [DataSet0,...,DataSetN]]
    #    0 - parameter value for percent increase in T
    #    1 - sim estimates of va average and attractor value
    #    2 - mc model intersection terms for va=ve
    #    3 - curve data to plot fitness gains vs changes in percent T
    #    4 - mc model for second set of plots with attractors and v estimates
    #    5 - va - vc estimates for sim runs
    #    6 - histogram data
    
    figData = dict.fromkeys(vperc)
    dataset_keys = ['T_perc','sim_avg','mc_int_pts',
                    'mc_curves','mc_model','v_est','hist_data']
    for key in figData.keys():
        figData[key] = dict(zip(dataset_keys, [[]]*7))
        
    # Now we loop through each file, get the data sets for entry 1 and entry 2.
    # However, we note that entry 2 only needs to be calculate once, since the 
    # MC model doesn't change across different ve sizes.
    for ii in range(nFiles):
        
        # get the current evoSim object. The evoSim object has the mc model, 
        # and output files needed for the figure data.
        evoFile = os.path.join(figSetup['workDir'],dataFiles['sim_snapshot'][ii]) 
        print("Processing: %s" % (evoFile))
        
        # get the current panel info and save the T-percent change 
        crntKey   = dataFiles['ve_percent'][ii]
        
        # ---------------------------
        # Data Collection
        # ---------------------------
        
        # 0. Save the current value for precent increase in T
        #    Note: this percentage is the starting value of ve/(va=vc)
        #
        # Usage: figDataSet[key][datakey][idx]
        #
        if figData[crntKey]['T_perc'] == []:
            figData[crntKey]['T_perc'] = [dataFiles['T_percent'][ii]]
        else:
            figData[crntKey]['T_perc'].append(dataFiles['T_percent'][ii])
        
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
        figData[crntKey]['sim_avg'] = figfun.get_simData(evoFile,figData[crntKey]['sim_avg'])
    
        # 2. Returns fine grid arrays to calculate fitness changes vs T percent 
        # change for the shift in va=ve intersection. This is just the MC model
        # data with key info extracted.
        # - ib: va=ve state
        # - bi: b-term of va=ve state
        # - ve: vE as percentage of va=vc
        # - fc: fitness difference from va=vc to va=ve
        # - ro: rho of mc model
        # - bs: b-term of attractor, va=vc
        # - vs: va at attractor
        # - ds: d-term at attractor (same at va=ve if d constant) 
        #
        # Usage: figDataSet[key][datakey][varkey][idx]
        #
        figData[crntKey]['mc_int_pts'] = figfun.get_mcModel_VaVeIntersect(evoFile,figData[crntKey]['mc_int_pts'])
        
        # 3. Returns arrays to plot curve of fitness gains vs territory size increases
        # - ib: state for va=ve intersection for curve defined by varying T
        # - bi: b-term for va=ve intersection for curve defined by varying T
        # - vp: vE as a percentage of va=vc intersection. Note latter changes
        #       with adjustments to T
        # - tv: T values for MC models derived from Tperc list
        # - fc: fitness change caused from shifting the va=ve intersection after
        #
        # Usage: figDataSet[key][datakey][varky] 
        # 
        if (figData[crntKey]['mc_curves']==[]) and (dataFiles['T_percent'][ii] == 100):
            pfixSolve = 3    # Use sel. coeff as pfix solution of v's
            figData[crntKey]['mc_curves'] = figfun.get_mcModel_VaVeIntersect_curveVaryT(evoFile,TvalsPerc,pfixSolve)
        
        # 4. get corresponding mc model
        # - includes all mc model parameters 
        #
        # Usage: figDataSet[key][datakey][idx]
        #
        if figData[crntKey]['mc_model'] == []:
            figData[crntKey]['mc_model'] = [figfun.get_mcModelFromEvoSim(evoFile)]
        else:
            figData[crntKey]['mc_model'].append(figfun.get_mcModelFromEvoSim(evoFile))
        
        # 5. get VaVcEstimates
        # - list with two elements: [states, vaEst], [states, vcEst]
        #
        # Usage: figDataSet[key][datakey][idx][varky]
        # varky: ['vaEst', 'vcEst']
        #
        if figData[crntKey]['v_est'] == []:
            figData[crntKey]['v_est'] = [figfun.get_effectiveVaVc(evoFile)]
        else:
            figData[crntKey]['v_est'].append(figfun.get_effectiveVaVc(evoFile))
    
        # 6. get sojourn times in states
        # - states, with weights for sojourn times
        #
        # Usage: figDataSet[key][datakey][idx][varid]
        # varid: [idxGrp,idxCnt]
        #
        if figData[crntKey]['hist_data'] == []:
            figData[crntKey]['hist_data'] = [figfun.get_stateDataForHist(evoFile)]
        else:
            figData[crntKey]['hist_data'].append(figfun.get_stateDataForHist(evoFile))
            
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
    saveFigFilename = os.path.join(figSetup['figSavePath'],figSetup['saveFigFile'])
    create_fitnessGainVsTincrFig(figDatSet,saveFigFilename,'taxis')
    
if __name__ == "__main__":
    main()
    
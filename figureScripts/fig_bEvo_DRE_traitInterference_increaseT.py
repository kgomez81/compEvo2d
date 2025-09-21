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

def create_singleTraitInterferenceFig(figDataSet,figSaveName):
    # create_singleTraitInterferenceFig() generates the main figure showing the 
    # degree of interference between the relative and absolute fitness traits.
    # 
    
    vperc = list(figDataSet.keys())
    nv = len(vperc)
    
    fig,ax = plt.subplots(3,1,figsize=[5,12])
        
    figData = process_sim_data_for_plots(figDataSet)
    
    cmap = cm.get_cmap('tab20b')
    
    
    for idx, key in enumerate(vperc):
        
        # plot the fitness increase of the average va for each ve level
        if (idx==0):
            # print(figData[key]['ve_perc'])
            print(figData[key]['fit_chng_env'])
            # print(figData[key]['fit_chng_int'])
        ax[idx].scatter(figData[key]['ve_perc'], figData[key]['fit_chng_avg'],color=cmap(idx/nv*1.0),marker='o',label=r'$v_{E,0}$'+("=%s%%"%(key)))
        # ax.scatter(figData[key]['ve_perc'], figData[key]['fit_chng_env'],color=cmap(idx/nv*1.0),marker='+') #,label=r'$v_{E,0}$'+("=%s%%"%(key)))
        # ax.scatter(figData[key]['ve_perc'], figData[key]['fit_chng_int'],color=cmap(idx/nv*1.0),marker='.') #,label=r'$v_{E,0}$'+("=%s%%"%(key)))
        ax[idx].plot(figData[key]['ve_perc_crv'], figData[key]['fit_chng_crv'],c='blue',linestyle='-.',label='No Interf')
    
    
        ax[idx].set_ylabel('Change in Fitness',fontsize=20,labelpad=8)
        # ax.set_ylim(myYlims)
        # ax.set_xlim([0,100])
        
        ax[idx].set_xlabel(r'$v_E/v_a^*$',fontsize=20,labelpad=8)
        # yticks = list(ax.get_yticks())
        # yticks = [round(elmnt,1) for elmnt in yticks]
        # yticklabels = list(map(str,yticks))
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(yticklabels,fontsize=14)
        
        # ax.set_xticks([20*ii for ii in range(0,6)])
        # ax.set_xticklabels([("%d%%" % (20*ii)) for ii in range(0,6)],fontsize=14)
        
        ax[idx].legend(fontsize=20)

    # ax.text(1,-0.16,r'$T=T_0 \times 1,5,100$',fontsize=20)    
    
    plt.tight_layout()
    
    figSaveName = figSaveName.replace('.pdf','_rho1.pdf')
    fig.savefig(figSaveName,bbox_inches='tight')
    
    return None

# --------------------------------------------------------------------------

def process_sim_data_for_plots(figDataSet):
    # function use to extract the specific sim data needed for the plot. Here
    # we organize the data retreived from get_figData. This is the data needed 
    # 
    # figDataSet contents
    # 1. 'i_avg','b_avg','v_avg','ve_perc','fit_chng','b_star','dTerm'
    # 2. 'ib','bi','ve','ve_perc','fit_chng','rho','b_star','v_star','dTerm'
    
    plot_types = list(figDataSet.keys())
    
    figData = dict.fromkeys(plot_types)
    
    dataKeys = ['ve_perc','fit_chng_avg','fit_chng_env','fit_chng_int','ve_perc_crv','fit_chng_crv']

    # each key here is an initial vE percent (eg. 50, 75, ... )
    for idx, key in enumerate(plot_types):
        
        vePercent  = []
        avgFitChng = []
        envFitChng = []
        intFitChng = []
        
        figData[key] = dict.fromkeys(dataKeys)
        
        # part 0 - get reference b-terms for starting T-value
        dRef    = figDataSet[key][0]['dTerm'][0]
        bAvgRef = figDataSet[key][0]['b_avg'][0]
        bEnvRef = figDataSet[key][1]['bi'][0]
        bIntRef = figDataSet[key][0]['b_star'][0]
        
        for ii in range(1,3):
            # Part 1 - plots for different T-values
            bAvgNew = figDataSet[key][0]['b_avg'][ii]
            bEnvNew = figDataSet[key][1]['bi'][ii]
            bIntNew = figDataSet[key][0]['b_star'][ii]
            
            vePercent.append(figDataSet[key][0]['ve_perc'][ii])
            avgFitChng.append(lmfun.get_b_SelectionCoeff(bAvgRef,bAvgNew,dRef))
            envFitChng.append(lmfun.get_b_SelectionCoeff(bEnvRef,bEnvNew,dRef))
            intFitChng.append(lmfun.get_b_SelectionCoeff(bIntRef,bIntNew,dRef))
        
        figData[key]['ve_perc'] = vePercent
        figData[key]['fit_chng_avg'] = avgFitChng
        figData[key]['fit_chng_env'] = envFitChng
        figData[key]['fit_chng_int'] = intFitChng
        figData[key]['ve_perc_crv'] = 100*np.asarray(figDataSet[key][2]['vEperc'])
        figData[key]['fit_chng_crv']= 100*np.asarray(figDataSet[key][2]['fitChng'])
    
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
    vperc = np.unique(dataFiles['ve_percent'].values)
    
    TpercMin = np.min(dataFiles['T_percent'].values)
    deltaT   = 1.0*(np.max(dataFiles['T_percent'].values) - TpercMin)
    nSample  = 10
    
    TvalsPerc = [TpercMin+deltaT*ii/10 for ii in range(nSample+1)]
    
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
    
    figData = dict.fromkeys(vperc)
    for key in figData.keys():
        figData[key] = [[],[],[],[]]
        
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
        crntKey = dataFiles['ve_percent'][ii]
        cnrtTperc = dataFiles['T_percent'][ii]
        
        # calculate the 1st entry of data set for current file and append
        # to respective dictionary entry for panel. 
        figData[crntKey][0] = figfun.get_simData(evoFile,figData[crntKey][0])
    
        # check if the mc model still hasn't been populated for this panel
        # if it has, then skip appending the data
        figData[crntKey][1] = figfun.get_mcModelVaVeIntersect(evoFile,figData[crntKey][1])
        
        # fine plot of fitness gains from shift in va-ve intersection due to 
        # territory size increase
        if (figData[crntKey][2]==[]):
            figData[crntKey][2] = figfun.get_mcModelVaVeIntersect_fitChngT(evoFile,TvalsPerc)
            
        figData[crntKey][3].append(cnrtTperc)
            
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
    create_singleTraitInterferenceFig(figDatSet,os.path.join(figSetup['figSavePath'],figSetup['saveFigFile']))
    
if __name__ == "__main__":
    main()
    
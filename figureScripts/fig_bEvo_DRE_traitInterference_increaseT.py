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
# Small supporting functions
# --------------------------------------------------------------------------

def get_sortedSimRuns_ByTvals(sim_t_vals):
    """
    Sort runs based on T-percent values specified for a simulation run
    inputs:
        - sim_t_vals = list of indices for sorted T-percent values 
    """
    
    # get an integer list of T-percent values
    T_perc_int = list(map(int,sim_t_vals))
        
    return list(np.argsort(T_perc_int))

# --------------------------------------------------------------------------

def get_T_sampling(T_perc_arry,nSample,Tscale=None):
    """
    Create a list of T-values to sample T values, with an option to select
    a linear or log scale
    
    Inputs:
        - T_perc_arry = Array of T-values to form bounds for sampling
        - nSample     = number of points to use for sample
        - Tscale      = string as 'lin' (linear) or 'log' (log) scale
    """
    TpercMin = np.min(T_perc_arry)
    TpercMax = np.max(T_perc_arry)
    
    if (Tscale==None) or (Tscale=='lin'):
        pDelta = (TpercMax - TpercMin)/(nSample-1)
        TvalsPerc = [(TpercMin+ii*pDelta) for ii in range(nSample)]
    else:
        pDelta = np.log10(TpercMax/TpercMin)/(nSample-1)
        TvalsPerc = [TpercMin*10**(ii*pDelta) for ii in range(nSample)]
    
    return TvalsPerc

#%% ------------------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------------------

def get_scaleAndTicks_Percent(valmin,valmax,axisType=None):
    """
    Function generates tics and labels, but is specialized for axis with
    percentages.
    
    inputs:
        valmin = minimum value of data set (expect percentage 0-1 type)
        valmax = maximum value of data set (expect percentage 0-1 type)
        scaleType = percent scale type, either 'bound' or 'multiples'
    outputs:
        valset = dictionary with tics and labels, along with scaling for labels
    """
    # notes: for the two types scales, we want the following setup
    # 1. for bounded, we want to choose either 5, 10, 25 percent increments
    #    depending on how many tics we get with each. Ideally, we'd want 3-5
    #    tics
    # 2. for the unbounded, we want tics based on multiples of 1
    
    valset = dict.fromkeys(['bnds','tics','lbls','scale','name'])
        
    if (axisType == 'vaxis') or (axisType == None):
        # these are the desired tic spacings
        ticset = [0.05,0.10,0.25]
        
        # select the choice that grants about 3 tics
        ticcnt = list(map(lambda x: (valmax-valmin)/x,ticset))
        ticopt = np.argmin(map(lambda x: abs(x-3),ticcnt))
        ntic   = np.ceil(ticcnt[ticopt])
        dtic   = ticset[ticopt]
        
        # find the first tick
        tic1 = valmin - np.mod(valmin,dtic)
        tics = [tic1+ii*dtic for ii in range(ntic)]
        tlbl = [("%d" % (tic)) for tic in tics]
        valname = '$v_E/v_a^*$'
        
    elif (axisType == 'taxis') or (axisType == None):
        # for this type, we hard code tics
        ticmult = [1,5,100]
        tics = [np.log10(tic) for tic in ticmult]
        tlbl = [str(tic)+'T' for tic in ticmult]
        valname = 'Multiples of Reference T ($log_{10}$)'
    
    # add info to tic dictionary
    valset['bnds'] = [np.min(tics),np.max(tics)]
    valset['tics'] = tics
    valset['lbls'] = tlbl
    valset['name'] = valname
    
    return valset

# --------------------------------------------------------------------------

def get_scaleAndTicks(valmin,valmax,valname,ntic=None):
    """
    function to select a plotting scale
    inputs:
        valmin = minimum value of data set
        valmax = maximum value of data set
        valname = name of axis to group with plotting parameters
        ntic = desired number of ticks (-1)
    outputs:
        valset = dictionary with tics and labels, along with scaling for labels
    """ 
    valset = dict.fromkeys(['bnds','tics','lbls','scale','name'])
    
    # find the largest power of 10 that is less than valmax, 
    # this will be your scale
    val_scale = 10**np.floor(np.log10(valmax))
    
    # define a max val rescaled accordingly
    valmax_scale = valmax/val_scale
    
    # now define increments based on the rescaled max val
    if (ntic == None) or (ntic < 0):
        ntic = 5
    dval_scale = valmax_scale/ntic*1.0
    
    # Want increment with 1 decimal place in spacing in rescaled land
    # and tic increments derived from scaling back those increments
    dval_scale = np.ceil(10.0*dval_scale)/10.0
    dtic = dval_scale*val_scale
    
    # create tics first
    tics = [ii*dtic for ii in range(ntic)]
    
    # now create the tic labels
    tlbl = [("%.1f" % (ii*dval_scale)) for ii in range(ntic)] 
    
    # save all of the info to the output dictionary
    dbnds = 0.05*(valmax-valmin)
    valset['bnds'] = [valmin-dbnds,valmax+dbnds]
    valset['tics'] = tics
    valset['lbls'] = tlbl
    valset['scale'] = val_scale
    valset['name'] = valname
    
    return valset

# --------------------------------------------------------------------------

def get_xlims(figData,xAxisType):
    """
    Function builds dictionary with x-axis plotting parameters. These are 
    specified for each initial vE value.
    
    Inputs:
        figData   -  x/y data from simulation and mc models
        xAxisType -  string indicating x-axis is of type vE as a percentage of 
                     the vb=vc value or a percent increase in T relative to a 
                     reference T value.
                     either: 'vaxis' or 'taxis'
    Outputs
        dictionary with panel x-axis limits, ticks, and labels 
    """
    
    # OLD SETTINGS FOR PRIOR VERSION OF PLOT
    # # xtick settings (change this if sim parameters change)
    # # NOTE: this is fine for the current selection of T sampling, but 
    # #       would need to change with Tperc in sim runs
    # xticDat = {'vals':{}, 'lbls': {}}
    # xticDat['vals'][0] = [35+5*ii for ii in range(4)]
    # xticDat['vals'][1] = [55+5*ii for ii in range(5)]
    # xticDat['vals'][2] = [70+10*ii for ii in range(4)]
    
    # get keys and subkeys of figData
    keys    = list(figData.keys())
    subkeys = list(figData[keys[0]].keys())
    
    xLimPar = dict.fromkeys(subkeys)
    xDatLbls = get_axisDataNames('x', xAxisType)
    
    # loop through the vE sets for each panel and generate the x-axis settings
    for subkey in subkeys:
        
        # Find the min and maximum values across a panel group
        xmax = 0
        xmin = 0
    
        for key in keys:
            for xdat in xDatLbls:
                xmax = np.max([xmax, np.max(figData[key][subkey][xdat])])
                xmin = np.min([xmin, np.min(figData[key][subkey][xdat])])
        
        # now use the min/max across the panel data to define tics for 
        # that axis, and also pass the name of the variable group 
        if (xAxisType == 'vaxis'):
            xLimPar[subkey] = get_scaleAndTicks_Percent(xmin,xmax,xAxisType)
        elif (xAxisType == 'taxis'):
            xLimPar[subkey] = get_scaleAndTicks_Percent(xmin,xmax,xAxisType)
            
        # add vE value for annotations
        xLimPar[subkey]['vE0'] = "%d%%" % (subkey)
    
    return xLimPar

# --------------------------------------------------------------------------

def get_ylims(figData,yAxisType):
    """
    Function builds dictionary with y-axis plotting parameters. These are 
    specified for each of the 
    
    Inputs
        figData   -  x/y data from simulation and mc models
        yAxisType -  string indicating fitness change or percent for y-axis 
                     either: 'fit_chng' or 'fit_chng_perc'
    Outputs
        dictionary with panel y-axis limits, ticks, and labels 
    """
    
    # get keys and subkeys of figData
    keys    = list(figData.keys())
    subkeys = list(figData[keys[0]].keys())
    
    yLimPar = dict.fromkeys(keys)
    yDatLbls = get_axisDataNames('y', yAxisType)
    
    # loop through the panels and generate the y-axis settings
    for key in keys:
        
        # Find the min and maximum values across a panel group
        ymax = 0
        ymin = 0
    
        for subkey in subkeys:
            for ydat in yDatLbls:
                ymax = np.max([ymax, np.max(figData[key][subkey][ydat])])
                ymin = np.min([ymin, np.min(figData[key][subkey][ydat])])
        
        # now use the min/max across the panel data to define tics for 
        # that axis, and also pass the name of the variable group 
        if (yAxisType == 'fit_chng'):
            yname = 'Change in Fitness'
        elif (yAxisType == 'fit_chng_perc'):
            yname = 'Change in Fitness (Perc.)'
        yLimPar[key] = get_scaleAndTicks(ymin,ymax,yname,ntic=5)
        
        # add rho for annotations
        mc_model = figData[key][subkeys[0]]['mc_model']
        yLimPar[key]['rho'] = "%.2f" % (mc_model.calculate_evoRho())
                
    return yLimPar

# --------------------------------------------------------------------------

def get_xlim_ylim_flags(idx,idy):
    """
    This function checks which axis require tic labels across the panels
    included in the main figure.
    
    Inputs:
        - idx = panel row for each rho value sampled in simulation
        - idy = panel column for each vE initial value used 
    Outputs:
        - dictionary with x/y entries for specific cases 
    """
    
    return {'x':(idx==2),'y':(idy==0),'lgd': (idx==0) and (idy==2)}

# --------------------------------------------------------------------------

def get_axisDataNames(axis=None, axisType=None):
    """
    Simple function to return the variable names that will be plotted for 
    specified axis
    
    Inputs:
        - axis = x or y axis
        - axisType = Name of data group to plot
                     xaxis is either: 'vaxis' or 'taxis'
                     yaxis is either: 'fit_chng' or 'fit_chng_perc'
    Outputs:
        - dataNames = list of strings with data to plot for axis specified
    """
    dataNames = None
    
    # x-axis names
    if axisType == 'vaxis' and axis == 'x':
        dataNames = ['ve_perc_init','ve_perc_init','ve_perc_crv']
    elif (axisType == 'taxis' or axisType == None) and axis == 'x':
        dataNames = ['T_perc_chng','T_perc_chng','T_perc_crv']
        
    # y-axis names
    elif axisType == 'fit_chng_perc' and axis == 'y':
        dataNames = ['fit_chng_avg_perc','fit_chng_int_perc','fit_chng_crv_perc']
    elif (axisType == 'fit_chng' or axisType == None) and axis ==  'y':
        dataNames = ['fit_chng_avg','fit_chng_int','fit_chng_crv']
    
    return dataNames

# --------------------------------------------------------------------------

def get_annotationCoordinates(xbnds,ybnds,atype):
    """
    Function to calculate the placement of annotations in panels
    """
    
    if atype == 'rho':
        yval = 0.95*np.diff(ybnds) + ybnds[0]
        xval = 0.60*np.diff(xbnds) + xbnds[0]
    elif atype == 'vE0':
        yval = 0.88*np.diff(ybnds) + ybnds[0]
        xval = 0.60*np.diff(xbnds) + xbnds[0]
    elif atype == 'pnl':
        yval = 0.95*np.diff(ybnds) + ybnds[0]
        xval = 0.03*np.diff(xbnds) + xbnds[0]
    
    return [xval,yval]

# --------------------------------------------------------------------------

def get_panelChar(idx,idy):
    """
    Function to get a panel desigation for plots
    """    
    
    return chr(65 + idy + 3*idx)

#%% ------------------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------------------

def plot_fitnessGainVsTincr(figDataSet,figSaveName,xAxisType=None,yAxisType=None):
    """
    Generates plot showing fitness increase vs increases to T
    Inputs:
        - figDataSet, contains all simulation and mc model data
        - figSaveName
        - xAxisType, string indicating vE percent or log(T/T0) x-axis
                     either: 'vaxis' or 'taxis'
        - yAxisType, string indicating fitness change or percent for y-axis 
                     either: 'fit_chng' or 'fit_chng_perc'
    Outputs:
        - figure showing fitness increases/decreases vs percent change in T
    """ 
    
    # Create main dictionary to store figure parameters
    myFig = dict.fromkeys(['data','axes'])
    
    # ---------------------------------------------------------------------     
    # Part I - process the simulation data 
    # ---------------------------------------------------------------------
    
    # Extract the sim and mc model data for the figure
    figData = get_simDataForPlots(figDataSet,'fitChngPlts')
    
    # Select data to plot, which will be used in loops
    myFig['data'] = {'x':None,'y':None}
    myFig['data']['x'] = get_axisDataNames('x',xAxisType)
    myFig['data']['y'] = get_axisDataNames('y',yAxisType)
        
    # ---------------------------------------------------------------------
    # Part II - get the figure/panel parameters and select data
    # ---------------------------------------------------------------------
    
    # Select params for plots 
    myFig['axes'] = {'x':None,'y':None}
    myFig['axes']['x'] = get_xlims(figData,xAxisType)
    myFig['axes']['y'] = get_ylims(figData,yAxisType)
    
    # ---------------------------------------------------------------------
    # Part III - generate the figure
    # ---------------------------------------------------------------------
    
    # setup the figure for each of the panels
    fig,ax  = plt.subplots(3,3,figsize=[13,13])   
    
    # loop through the panels data and plot it
    for idx,key in enumerate(figData.keys()):
        
        for idy,subkey in enumerate (reversed(figData[key].keys())):
            
            # plot horizontal line for reference 
            ax[idx,idy].plot(myFig['axes']['x'][subkey]['bnds'], [0]*len(myFig['axes']['x'][subkey]['bnds']) ,c='black',linestyle='--')
            
            # get the data names
            xdat = myFig['data']['x']
            ydat = myFig['data']['y']
            
            ax[idx,idy].scatter(figData[key][subkey][xdat[0]], figData[key][subkey][ydat[0]],color='black',marker='o',label='Imperfect Interference')
            ax[idx,idy].scatter(figData[key][subkey][xdat[1]], figData[key][subkey][ydat[1]],color='red',marker='o',facecolors='none',label='Perfect Interference')
            ax[idx,idy].plot(figData[key][subkey][xdat[2]], figData[key][subkey][ydat[2]],c='blue',linestyle='-',marker='o',markersize='2',label='No Interference')
            
            # bnds also needed for annotations)
            xbnds = myFig['axes']['x'][subkey]['bnds']
            ybnds = myFig['axes']['y'][key]['bnds']
            
            # get flags for tics and legends
            plotflags = get_xlim_ylim_flags(idx,idy)
            
            # set the x-axis parameters
            #print(myFig['axes']['x'])
            ax[idx,idy].set_xticks      (myFig['axes']['x'][subkey]['tics'])
            if (plotflags['x']):
                ax[idx,idy].set_xticklabels (myFig['axes']['x'][subkey]['lbls'],fontsize=14)
            else:
                ax[idx,idy].set_xticklabels (['']*len(myFig['axes']['x'][subkey]['lbls']),fontsize=14)
            ax[idx,idy].set_xlim        (myFig['axes']['x'][subkey]['bnds'])    
            
            # set the x-axis parameters
            #print(myFig['axes']['y'])
            ax[idx,idy].set_yticks      (myFig['axes']['y'][key]['tics'])
            if (plotflags['y']):
                ax[idx,idy].set_yticklabels (myFig['axes']['y'][key]['lbls'],fontsize=14)
            else:
                ax[idx,idy].set_yticklabels (['']*len(myFig['axes']['y'][key]['lbls']),fontsize=14)
            ax[idx,idy].set_ylim        (myFig['axes']['y'][key]['bnds'])    
            
            # add legend if needed
            if (plotflags['lgd']):
                ax[idx,idy].legend(fontsize=16,loc='center right')
            
            # set panel, rho and vE init
            xypos = get_annotationCoordinates(xbnds,ybnds,'pnl')
            annot = "(%s)" % (get_panelChar(idx,idy))
            ax[idx,idy].text(xypos[0],xypos[1],annot,fontsize=16)
            
            # set rho 
            xypos = get_annotationCoordinates(xbnds,ybnds,'rho')
            annot = "%s=%s" %(r'$\rho$',myFig['axes']['y'][key]['rho'])
            ax[idx,idy].text(xypos[0],xypos[1],annot,fontsize=16)
            
            # set vE init
            xypos = get_annotationCoordinates(xbnds,ybnds,'vE0')
            annot = "%s=%s" %(r'$vE/v^*$',myFig['axes']['x'][subkey]['vE0'])
            ax[idx,idy].text(xypos[0],xypos[1],annot,fontsize=16)
            
    
    
    # set axis super labels
    if xAxisType == 'vaxis':
        fig.supxlabel(r'$v_E/v_a^*$',y=-0.01,fontsize=16)
    else:
        fig.supxlabel(r'Multiples of Reference T ($log_{10}$)',y=0.01,fontsize=16)
    fig.supylabel('Change in Fitness',x=0.01,fontsize=16)
    
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    
    # ---------------------------------------------------------------------
    # Part IV - Save the figure
    # ---------------------------------------------------------------------
    
    plt.tight_layout()
    fig.savefig(figSaveName,bbox_inches='tight')
    
    return None

# --------------------------------------------------------------------------

def plot_allMcModels(figDataSet,savepath,basefigname):
    """
    Function to plot all of the mc models and sim data used for the 
    fitness change vs T increase plots
    """
    
    figData = get_simDataForMcPlots(figDataSet)  
    
    # loop through all of the mc models
    for key in figData.keys():
        for subkey in figData[key].keys():
            for tval in figData[key][subkey].keys():
                
                savename = basefigname.replace('.pdf',f'_pnl{key}_ve{subkey}_t{tval}.pdf')
                plot_McModels(figData[key][subkey][tval],os.path.join(savepath,savename))
    
    return None

# --------------------------------------------------------------------------

def plot_McModels(figDataMc,savename):
    
    # Figures to plot include the mean fitnes abs and relative
    # 1. MC model w/ histogram
    # 2. MC model w/ va estimates
    
    fig,ax = plt.subplots(2,1,figsize=[6,12])
    
    # MC model
    ax[0].scatter(figDataMc['mcMod']['abs']['ib'],figDataMc['mcMod']['abs']['v'],c='blue',label='vb')
    ax[0].scatter(figDataMc['mcMod']['rel']['ib'],figDataMc['mcMod']['rel']['v'],c='red',label='vc')
    ax[0].plot   (figDataMc['mcMod']['env']['ib'],figDataMc['mcMod']['env']['v'],c='black',label='vE')
    ax[0].set_xlabel('absolute fitness state')
    ax[0].set_ylabel('rate of adaptation')
    ax[0].legend()
    ax0t = ax[0].twinx()
    ax0t.hist(figDataMc['mcHist']['states'],weights=figDataMc['mcHist']['wghts'],alpha = 0.5, color= 'k')
    ax0t.set_yticks([])
    
    # Mean gamma over time (generations)
    ax[1].scatter(figDataMc['mcMod']['abs']['ib'],figDataMc['mcMod']['abs']['v'],c='blue',label='vb')
    ax[1].scatter(figDataMc['mcMod']['rel']['ib'],figDataMc['mcMod']['rel']['v'],c='red',label='vc')
    ax[1].plot   (figDataMc['mcMod']['env']['ib'],figDataMc['mcMod']['env']['v'],c='black',label='vE')
    
    ax[1].scatter(figDataMc['vEst']['abs']['ib'],figDataMc['vEst']['abs']['v'],facecolors='none', edgecolors='cyan',label='vbEst')
    ax[1].scatter(figDataMc['vEst']['rel']['ib'],figDataMc['vEst']['rel']['v'],facecolors='none', edgecolors='magenta',label='vcEst')
    
    ax[1].set_xlabel('absolute fitness state')
    ax[1].set_ylabel('rate of adaptation')
    ax[1].legend()
    
    # save figure in location where outputs are located
    fig.savefig(savename,bbox_inches='tight')
    plt.close()
    
    return None

#%% ------------------------------------------------------------------------
# Sim Data Functions 
# --------------------------------------------------------------------------

def get_simDataForPlots(figDataSet,fig_type):
    """
    Extracts and organizes sim and mc model data for the plots of fig_type. 
    
    Inputs:
        - figDataSet, struct with all processed sim and mc model data
        - fig_type, indicator for data selection
    Outputs:
        - figData, dictionary with data organized to plot of main figures 
          showing changes in fitness due to changes in T at various values 
          of vE.
    """
    
    # list of data included in outputs for plots 
    dataKeys = ['ve_perc_init','T_perc_chng',
                'fit_chng_avg','fit_chng_env','fit_chng_init',
                'fit_chng_avg_perc','fit_chng_env_perc','fit_chng_int_perc',
                've_perc_crv','T_perc_crv','fit_chng_crv','fit_chng_crv_perc']
    
    # build up the nested dictionary with plot data
    figData = {}
    
    for key in list(figDataSet.keys()):
        figData[key] = dict.fromkeys(figDataSet[key].keys())
        
        for subkey in list(figDataSet[key].keys()):
            
            # Get the init v0, which will be the max vp for the given subkey
            vp0 = np.max(figDataSet[key][subkey]['sim_avg']['vp'])
            
            vpi = [ vp0 ]    # initial vE as a percent of v* at vc=vb
            tpc = [   0 ]    # Territory size percent
            
            fca = [   0 ]    # fitness change due to change in average state
            fce = [   0 ]    # fitness change due to change in ve=va location
            fci = [   0 ]    # fitness change due to change in va=vc location
            
            fcap = [   0 ]    # fitness change due to change in average state
            fcep = [ 100 ]    # fitness change due to change in ve=va location
            fcip = [   0 ]    # fitness change due to change in va=vc location
            
            # First sort data set by increasing order of T-percent.
            figIdx = get_sortedSimRuns_ByTvals(figDataSet[key][subkey]['T_perc'])
            
            # Get reference b-term for starting T-value (1st index)
            id0 = figIdx[0]
            
            di0 = figDataSet[key][subkey]['sim_avg']['ds'][id0]     # d-term (doesn't change)
            ba0 = figDataSet[key][subkey]['sim_avg']['ba'][id0]     # b-term at average
            be0 = figDataSet[key][subkey]['mc_int_pts']['bi'][id0]  # b-term at va=ve
            bs0 = figDataSet[key][subkey]['mc_int_pts']['bs'][id0]  # b-term at va=vc
            
            for idx in figIdx[1:]:
                
                # Part 1 - get the current b-erms
                ba1 = figDataSet[key][subkey]['sim_avg']['ba'][idx]
                be1 = figDataSet[key][subkey]['mc_int_pts']['bi'][idx]
                bs1 = figDataSet[key][subkey]['mc_int_pts']['bs'][idx]
                
                vpi.append(figDataSet[key][subkey]['sim_avg']['vp'][idx])
                tpc.append(np.log10(figDataSet[key][subkey]['T_perc'][idx]/100))
                
                fca.append(lmfun.get_b_SelectionCoeff(ba0,ba1,di0))
                fce.append(lmfun.get_b_SelectionCoeff(be0,be1,di0))
                fci.append(lmfun.get_b_SelectionCoeff(bs0,bs1,di0))
                
                fcap.append(100*fca[-1]/fce[-1])
                fcep.append(100*fce[-1]/fce[-1])
                fcip.append(100*fci[-1]/fce[-1])
                
            # store the values 
            figData[key][subkey] = dict.fromkeys(dataKeys)
                
            figData[key][subkey]['ve_perc_init'] = np.asarray(vpi)
            figData[key][subkey]['T_perc_chng' ] = np.asarray(tpc)
            figData[key][subkey]['fit_chng_avg'] = np.asarray(fca)
            figData[key][subkey]['fit_chng_env'] = np.asarray(fce)
            figData[key][subkey]['fit_chng_int'] = np.asarray(fci)
            
            figData[key][subkey]['fit_chng_avg_perc'] = np.asarray(fcap)
            figData[key][subkey]['fit_chng_env_perc'] = np.asarray(fcep)
            figData[key][subkey]['fit_chng_int_perc'] = np.asarray(fcip)
            
            # Part 2 - get the fine grid array (crv=curves) to plot fitness change
            # as a runction of the ve percent or T change in percent
            figData[key][subkey]['ve_perc_crv' ] = 100*np.asarray(figDataSet[key][subkey]['mc_curves']['vp'])
            figData[key][subkey]['T_perc_crv'  ] = np.log10(np.asarray(figDataSet[key][subkey]['mc_curves']['tp']))
            figData[key][subkey]['fit_chng_crv'] = np.asarray(figDataSet[key][subkey]['mc_curves']['fc'])
            figData[key][subkey]['fit_chng_crv_perc'] = 100*np.ones(figData[key][subkey]['fit_chng_crv'].shape)
            
            # keep a copy of the mc model for the first T=100
            figData[key][subkey]['mc_models'] = figDataSet[key][subkey]['mc_model'][0]
    
    return figData

# --------------------------------------------------------------------------

def get_simDataForMcPlots(figDataSet):
    """
    Function to extract the specific sim data needed for the MC plots. Here
    we organize the data retreived from get_GroupedfigData, and build up the
    MC model data to get plots each of the runs.
    """
    
    
    figData = dict.fromkeys(figDataSet.keys())
    
    for key in figDataSet.keys():
        
        figData[key] = dict.fromkeys(figDataSet[key].keys())
        
        for subkey in figDataSet[key].keys():
            
            figData[key][subkey] = dict.fromkeys(figDataSet[key][subkey]['T_perc'])
            
            for idx, tval in enumerate(figData[key][subkey].keys()):
            
                # create output dictionary
                figData[key][subkey][tval] = {'mcMod': {'abs':[],'rel':[],'env':[]},'vEst':{'abs':[],'rel':[]},'mcHist':[],'rho':[]}
    
                # get the current mc model and store the state space
                crntMcModel = figDataSet[key][subkey]['mc_model'][idx]
                figData[key][subkey][tval]['mcMod']['abs'] = {'ib': crntMcModel.state_i, 'v': crntMcModel.va_i}
                figData[key][subkey][tval]['mcMod']['rel'] = {'ib': crntMcModel.state_i, 'v': crntMcModel.vc_i}
                figData[key][subkey][tval]['mcMod']['env'] = {'ib': crntMcModel.state_i, 'v': crntMcModel.ve_i}
                
                # gather the v estimates from the simuation data
                figData[key][subkey][tval]['vEst']['abs'] = {'ib': figDataSet[key][subkey]['v_est'][idx]['vaEst']['ix'], 
                                                             'v': figDataSet[key][subkey]['v_est'][idx]['vaEst']['vx']}
                figData[key][subkey][tval]['vEst']['rel'] = {'ib': figDataSet[key][subkey]['v_est'][idx]['vcEst']['ix'], 
                                                             'v': figDataSet[key][subkey]['v_est'][idx]['vcEst']['vx']}
                
                # histogram data
                figData[key][subkey][tval]['mcHist'] = figDataSet[key][subkey]['hist_data'][idx]
                
                # mc Model parameters
                figData[key][subkey][tval]['rho'] = crntMcModel.calculate_evoRho()
    
    return figData

# --------------------------------------------------------------------------

def get_GroupedfigData(figSetup,panelkey):
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
        evoFile = os.path.join(figSetup['workDir'],dataFiles['sim_snapshot'].iloc[ii]) 
        print("Processing: %s" % (evoFile))
        
        # get the current panel info and save the T-percent change 
        crntKey   = dataFiles['ve_percent'].iloc[ii]
        
        # ---------------------------
        # Data Collection
        # ---------------------------
        
        # 0. Save the current value for precent increase in T
        #    Note: this percentage is the starting value of ve/(va=vc)
        #
        # Usage: figDataSet[key][datakey][idx]
        #
        if figData[crntKey]['T_perc'] == []:
            figData[crntKey]['T_perc'] = [dataFiles['T_percent'].iloc[ii]]
        else:
            figData[crntKey]['T_perc'].append(dataFiles['T_percent'].iloc[ii])
        
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
        if (figData[crntKey]['mc_curves']==[]) and (dataFiles['T_percent'].iloc[ii] == 100):
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

#%% ------------------------------------------------------------------------
# Main function to run script with
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
    
    # store all panel data in one dictionary for access later
    figDataSet = dict.fromkeys(panel_set)
    
    # we process panel data seperately in case it needs to be plotted 
    # in differentfigures
    for panelkey in panel_set:
        
        savePickleData = figSetup['saveData'].replace('.pickle',f'_pnl{panelkey}.pickle')
        if not (os.path.exists(savePickleData)):
            # start timer
            tic = time.time()
            
            # get the date for the figure
            figDataSet[panelkey] = get_GroupedfigData(figSetup,panelkey)
                
            # save the data to a pickle file
            with open(savePickleData, 'wb') as file:
                # Serialize and write the variable to the file
                pickle.dump(figDataSet[panelkey], file)
                
            print(time.time()-tic)
    
        else:
            # load mcModel data
            with open(savePickleData, 'rb') as file:
                # Serialize and write the variable to the file
                figDataSet[panelkey] = pickle.load(file)
                
    # create the figure (3x3 panels for all data)
    saveFigFilename = os.path.join(figSetup['figSavePath'],figSetup['saveFigFile'])
    # plot_fitnessGainVsTincr(figDataSet,saveFigFilename,xAxisType='taxis',yAxisType='fit_chng')
    
    plot_allMcModels(figDataSet,figSetup['figSavePath'],figSetup['saveFigFile'])
    
if __name__ == "__main__":
    main()
    

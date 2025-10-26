# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:26:48 2024

@author: Kevin Gomez
"""
# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.LotteryModel import LM_functions as lmfun

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_factory as mcFac

#%% ------------------------------------------------------------------------
############################################################################
#                         Plot Helper Functions   
############################################################################
#
# Small functions to prep data for plots, or adjust axes limits and labels
#
# --------------------------------------------------------------------------

def getScatterData(X,Y,Z):
    
    x = []
    y = []
    z = []
    
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            
            # removed bad data
            xGood = not np.isnan(X[ii,jj]) and not np.isinf(X[ii,jj])
            yGood = not np.isnan(Y[ii,jj]) and not np.isinf(Y[ii,jj])
            zGood = not np.isnan(Z[ii,jj]) and not np.isinf(Z[ii,jj])
            
            if xGood and yGood and zGood:
                x = x + [ X[ii,jj] ]
                y = y + [ Y[ii,jj] ]
                z = z + [ Z[ii,jj] ]
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    return [x,y,z]

# --------------------------------------------------------------------------

def getScatterData_special(X,Y,Z):
    
    x = []
    y = []
    z = []
    
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            
            # removed bad data
            xGood = not np.isnan(X[ii,jj]) and not np.isinf(X[ii,jj])
            yGood = not np.isnan(Y[ii,jj]) and not np.isinf(Y[ii,jj])
            zGood = not np.isnan(Z[ii,jj]) and not np.isinf(Z[ii,jj])
            yMlim = (np.abs(Y[ii,jj])<=2)
            
            if xGood and yGood and zGood and yMlim:
                x = x + [ X[ii,jj] ]
                y = y + [ Y[ii,jj] ]
                z = z + [ Z[ii,jj] ]
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    return [x,y,z]

# --------------------------------------------------------------------------

def get_emptyPlotAxisLaels(lbls):
    # simple function to get empty labels for plots
    
    emptyLbls = ['' for ii in range(len(lbls))]
    
    return emptyLbls

#%% ------------------------------------------------------------------------
##########################################################################
#                  Sim data processing functions                         
##########################################################################
#
# Set of functions below help process sim data for plots. They are grouped
# here for re-use across multiple scripts.
#
# --------------------------------------------------------------------------

def get_mcModel_VaVeIntersect(evoSimFile,priorMcModelData=None):
    """
    Returns va-ve intersection with associated b-term and states
    inputs:
      - evoSimfFile - file with simulation snapshot object
      - priorMcModelData - prior data to append new data to (default None)
    outputs: 
      dict 
      - ib: va=ve state
      - bi: b-term of va=ve state
      - ve: vE as percentage of va=vc
      - fc: fitness difference from va=vc to va=ve
      - ro: rho of mc model
      - bs: b-term of attractor, va=vc
      - vs: va at attractor
      - ds: d-term at attractor (same at va=ve if d constant)
    """
    
    # load evoSim object to ge the mcModel
    evoSim = get_evoSnapshot(evoSimFile)    
    
    # get va=vc and va=vb intersection indices
    ie = int(np.max(evoSim.mcModel.get_va_ve_intersection_index()))
    ic = int(np.max(evoSim.mcModel.get_va_vc_intersection_index()))
    ro = evoSim.mcModel.calculate_evoRho()
    
    # get state, v, bi for va=ve and va=vc 
    ib = evoSim.mcModel.state_i[ie]  # ib=ie for b-evo models
    bi = evoSim.mcModel.bi     [ie]
    ve = evoSim.mcModel.va_i   [ie]  
    
    bs = evoSim.mcModel.bi     [ic]
    ds = evoSim.mcModel.di     [ic]
    vs = evoSim.mcModel.va_i   [ic]
    
    # ve=va as percent of attractor & fitness increases relative to latter
    vp = 100.0 * ve / vs
    fc = lmfun.get_b_SelectionCoeff(bs,bi,ds)
    
    # set data as outputs, each entry has to be a list to allow extension
    mykeys      = ['ib','bi','ve','vp','fc','ro','bs','vs','ds']
    mcIntData   = [[ib],[bi],[ve],[vp],[fc],[ro],[bs],[vs],[ds]]
    
    if (priorMcModelData==[]) or (priorMcModelData==None):
        mcModelData = dict(zip(mykeys,mcIntData))
    else:
        for ikey, key in enumerate(mykeys):
            priorMcModelData[key].extend(mcIntData[ikey])
        mcModelData = priorMcModelData
        
    return mcModelData

# --------------------------------------------------------------------------

def get_mcModel_VaVeIntersect_curveFixedT(evoSimFile):
    """
    Returns curve data representing for va=ve intersections from varying ve.
    inputs:
     - evoSimFile - sim pickly snapshot
    outputs: 
     dict part 1 fixVals
     - ro: rho value of the mc model
     - bs: b-term at attractor
     - vs: va at attractor
     - ds: d-term at attractor
    
     dict part 2 arrVals
     - ib: states at va=ve intersections, varying T
     - bi: b-terms at va=ve intersections, varying T
     - ve: intersection vE as T varies
     - vp: intersection vE as percentage of va=va attractor, vary's with T
     - fc: fitness change from change in intersection va=ve, due to change 
           in T, relative to base va=ve value (T=100%)
    """
        
    # load evoSim object to ge the mcModel
    evoSim = get_evoSnapshot(evoSimFile)    
    
    # get va=vc and va=vb intersection indices
    ic = int(np.max(evoSim.mcModel.get_va_vc_intersection_index()))
    ro = evoSim.mcModel.calculate_evoRho()
    
    # Store some reference values for calculations
    bs = evoSim.mcModel.bi[ic]
    vs = evoSim.mcModel.va_i[ic]
    ds = evoSim.mcModel.di[ic]
    
    # now get all of the va's, bi above the intersection
    ib = evoSim.mcModel.state_i[ic:]
    bi = evoSim.mcModel.bi     [ic:]
    ve = evoSim.mcModel.va_i   [ic:]  # value of ve when ve=va
    
    # clean up the data for only non negative va
    ib = ib[ve>0]
    bi = bi[ve>0]
    ve = ve[ve>0]
    
    # calculate fitness difference along ve(=va) relative to attractor, and 
    # ve percentage of va(=vc) intersection.
    fc = np.zeros(ve.shape)
    vp = np.zeros(ve.shape)
    
    for ii in range(bi.shape[0]):
        vp[ii] = 100.0 * ve[ii] / vs
        fc[ii] = lmfun.get_b_SelectionCoeff(bs,bi[ii],ds)
    
    # define data for outputs
    mykeysFix = ['ro','bs','vs','ds']
    mcDataFix = [ ro , bs , vs , ds ]
    
    mykeysArr = ['ib','bi','ve','vp','fc']
    mcDataArr = [ ib , bi , ve , vp , fc ]
    
    # should only be calculated once, per parameter set
    return {'fixVals': dict(zip(mykeysFix,mcDataFix)),'arrVals': dict(zip(mykeysArr,mcDataArr))}

# --------------------------------------------------------------------------

def get_mcModel_VaVeIntersect_curveVaryT(evoSimFile,Tperc):
    """
    Returns curve data for va=ve from changing T 
    inputs: 
      - evoSimFile: file with simulation snapshot object 
      - Tperc:      Increase in T by percentages, e.g. [100, ..., 1000], 
                    as integer values
    outputs:
      dict - ['ib','bi','vp','fc']
      - ib: state for va=ve intersection for curve defined by varying T
      - bi: b-term for va=ve intersection for curve defined by varying T
      - vp: vE as a percentage of va=vc intersection. Note latter changes
            with adjustments to T
      - fc: fitness change caused from shifting the va=ve intersection after
            adjusting T. 
    """
    
    # load evoSim object to get the mcModel with a starting vE value 
    evoSim = get_evoSnapshot(evoSimFile)    
    
    # simInit struct as a baseline for new copies w/ adjusted territory sizes    
    simInitRef = evoSim.simInit
    
    # store the base value of b-term for va=vE
    ie = int(np.max(simInitRef.mcModel.get_va_ve_intersection_index()))
    ic = int(np.max(simInitRef.mcModel.get_va_vc_intersection_index()))
    
    # reference b, d, vE, v-intersection, and T size for base model
    biRef = simInitRef.mcModel.bi[ie]
    diRef = simInitRef.mcModel.di[ie]
    vERef = simInitRef.mcModel.va_i[ie]
    
    vIRef = simInitRef.mcModel.va_i[ic]
    T0Ref = simInitRef.mcModel.params['T']

    # create list to store computed values
    ibVals = [ ie          ]
    biVals = [ biRef       ]
    veVals = [ vERef/vIRef ]
    fcVals = [ 0           ]
    
    # now loop through all of the TvalsPerc and calculate a fitness change due
    # to the increase of the territory size
    for idx, Tp in enumerate(Tperc):
        if idx == 0:
            # first values have already been initialized
            continue
        else:
            # Adjust Territory size recalculate MC model with changes 
            #  Note: there might be an easier way to do this directly by just 
            #        scaling the va and vc values directly, but T also alters 
            #        pfix values, which would be adjusted in non-trivial way
            simInitRef.params['T'] = (Tp/100.0) * T0Ref
            simInitRef.recaculate_mcModel()
            
            # append the b-terms, ve percents, and fitness changes
            ie = int(np.max(simInitRef.mcModel.get_va_ve_intersection_index()))
            ic = int(np.max(simInitRef.mcModel.get_va_vc_intersection_index()))
            
            # calculate the current vE as percent of current intersection
            vE_perc = simInitRef.mcModel.va_i[ie]/simInitRef.mcModel.va_i[ic]
            bE_crnt = simInitRef.mcModel.bi[ie]
            fc_crnt = lmfun.get_b_SelectionCoeff(biRef,bE_crnt,diRef)
            
            ibVals.append( ie      )
            biVals.append( bE_crnt )
            veVals.append( vE_perc )
            fcVals.append( fc_crnt )
    
    # set data as outputs
    mykeys      = ['ib','bi','vp','fc']
    mcEnvData   = [ ibVals, biVals, veVals, fcVals ]
    
    # should only be calculated once, per parameter     
    return dict(zip(mykeys,mcEnvData))

# --------------------------------------------------------------------------
def get_simData(evoSimFile,priorFigData):
    """
    Returns mean state for sim run, and comparison w/ theoretical va=vc
    inputs:
     - evoSimFile:   file with simulation snapshot object 
     - priorFigData: prior data to append new data to (default None)
    Outputs - Dictionary with fields:
     - ia: average state 
     - ba: b-term at average state
     - vp: ve as percent of va=vc value for mc model
     - fc: fitness difference attractor thry and average state
     - bs: b-term at attractor
     - ds: d-term at attractor
    """
    
    # Get the evoSim object
    evoSim = get_evoSnapshot(evoSimFile)
    
    # load stat file with sampled data (paths need to be adjusted)
    statsFile = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.outputStatsFile)[1] )
    data      = pd.read_csv(statsFile)
    
    # get index of average state and attractor
    ia = int(np.mean(data['mean_b_idx'].values))    
    ic = int(np.max(evoSim.mcModel.get_va_vc_intersection_index()))
    ro = evoSim.mcModel.calculate_evoRho()
    
    # calculate assocated terms
    ba = evoSim.mcModel.bi[ia]
    bs = evoSim.mcModel.bi[ic]
    ds = evoSim.mcModel.di[ic]
    
    vs = evoSim.mcModel.va_i[ic]
    ve = evoSim.mcModel.ve_i[ic]
    vp = 100.0 * ve / vs
    
    # fitness difference from attractor to average state
    fc = lmfun.get_b_SelectionCoeff(bs,ba,ds)

    # group the new data
    mykeys      = ['ia','ba','vp','fc','bs','ds','ro']
    new_data    = [[ia],[ba],[vp],[fc],[bs],[ds],[ro]]
    
    if (priorFigData==[]):
        figData = dict(zip(mykeys,new_data))
    else:
        for ikey, key in enumerate(mykeys):
            priorFigData[key].extend(new_data[ikey])
        figData = priorFigData
    
    return figData

# --------------------------------------------------------------------------

def get_rateOfEnvironmentalChange_samples(tIter,biFit,dterm,envJmp):
    # function takes an array of realized fitness changes due the environment
    # and creates estimates of the rate of environmental change.
    #
    # Assumptions: 
    # 1. population is homogeneous in b-mutations, i.e. bmin=bmax, for all t
    # 2. environmental changes occur at non-trivial rate.
    # 3. no d-evolution
    #
    # Inputs:
    # tIter - time in terms of model iterations
    # biFit - b-fitness of population
    # dterm - d-term of the population (assumed no mutations in d trait)
    # envJmp- array with cumulative count of jumps from env change
    
    # first get the jumps in fitness from environment 
    idx2 = np.where(np.diff(envJmp)>0)[0]+1  # index of jumps
    
    # now 0 with other index of jumps to get duration with no env changes
    idx1     = np.where(np.diff(envJmp)>0)[0]+1
    idx1[1:] = idx1[0:-1]
    idx1[0]  = 0

    # list of start to jump values
    idxJmp = [idx1,idx2]

    biJmp  = [biFit[idxJmp[0]],biFit[idxJmp[1]]]
    tiJmp  = [tIter[idxJmp[0]],tIter[idxJmp[1]]]
    nJmps  = len(idxJmp[0])
    
    # calculate tau (iterations per generation)
    tau = 1.0/(dterm-1.0)
    
    # create array to store vE estimates
    vEi = np.zeros(idxJmp[0].shape)
    
    # get vE estimates from breaks
    for ii in range(nJmps):
        # change in fitness
        delta_si = lmfun.get_b_SelectionCoeff(biJmp[0][ii],biJmp[1][ii],dterm)
        
        # estiamte of rate of environmental change (time only), in generations
        T_Ei     = (tiJmp[1][ii] - tiJmp[0][ii]) / tau
        
        # estimate of vE for current jump.
        vEi[ii]  = delta_si/T_Ei
        
    return [vEi,idxJmp[1]]

#------------------------------------------------------------------------------
    
def calculate_RateOfAdapt_estimates(evoSim):
    """
    Returns estimates for rate of adaptation from sim run logs
    inputs:
     - evoSim: simulation pickle snapshot
    outputs:
     - vaEstimates: array of estimates for va; states given by indices so that
                    dimension of vaEstimates match MC model bi array.
    """
    
    # initialize all estimates to zero
    vaEstimates = np.zeros(evoSim.mcModel.va_i.shape)

    # load the data from the output file, w/ headers
    # fitness_state
    # sojourn_time
    # sojourn_kappa
    # adapt_counter
    data = pd.read_csv(evoSim.get_adaptiveEventsLogFilename())
    
    # process data to calculate the estimates as follows:
    #   1. T_i = iterations spend in state at state k
    #   2. T_bar_k = (1/n) * sum_(T_i for state k)
    #   3. va_iter = sa_k / T_bar_k
    #   4. va_gen = tau * va_iter
    
    # loop through the states with data
    states = list(map(int,np.sort(np.unique(data['fitness_state'].values))))
    for ii in states:
        tempData = data.query('fitness_state == '+str(ii)+' and sojourn_kappa < 0.4')
        # query for the entries if state ii
        if (len(tempData)>0):
            Tbar = np.sum(tempData['sojourn_time'].values)/len(tempData)
            sa = evoSim.mcModel.sa_i[ii]
            tau = 1/(evoSim.mcModel.di[ii]-1)
            vaEstimates[ii] = sa/Tbar*tau
            
    return vaEstimates

#------------------------------------------------------------------------------
    
def get_estimateRateOfAdaptFromSim(data,fitType,mcModel):
    """
    Return estimates for rate of adaptation w/ states.
      Note: here we make the assumption that the index is the state space.
            this could change if using dEvo model!
    inputs:
     - data: panda's data frame with sim log data.
     - fitType: string to select abs or rel fitness estimates
     - mcModel: MC model object
    outputs:
     dict
     - ix: states with nonzero estimates for rate adaptation
     - vx: rate of adaptation estimates; method inputs determine abs/rel type
    """
    
    # data from adaptive event log files should have following headers:
    #
    # fitness_state - abs or rel fitness state
    # sojourn_time  - time spent in the fitness state
    # sojourn_kappa - sojourn time adjustment 
    # adapt_counter - counter for number of adaptive events (-1)
    # crnt_abs_state - zero if abs log, and abs fitness state for rel log
    #
    
    # process data to calculate the estimates as follows:
    #   1. T_i = iterations spend in state at state k
    #   2. T_bar_k = (1/n) * sum_(T_i for state k)
    #   3. va_iter = sa_k / T_bar_k
    #   4. va_gen = tau * va_iter
    
    # loop through the states with data
    if (fitType == 'abs'):
        fit_state_key = 'fitness_state'
    elif (fitType == 'rel'):
        fit_state_key = 'crnt_abs_fit'
    
    states = list(map(int,np.sort(np.unique(data[fit_state_key].values))))
    idxStates = np.array(states)
    vaEstimates = np.zeros(idxStates.shape)
    
    for ii, state in enumerate(states):
        
        # query for the entries if state ii
        tempData = data.query(fit_state_key + ' == '+str(state)+' and sojourn_kappa < 0.4')
        
        # if date was found meeting the criteria, then calculate an estimate of v
        if (len(tempData)>0):
            Tbar = np.sum(tempData['sojourn_time'].values)/len(tempData)
            sa = mcModel.sa_i[state]
            tau = 1/(mcModel.di[state]-1)
            vaEstimates[ii] = sa/Tbar*tau
            
    # return dictionary with states and estimates
    return {'ix': idxStates,'vx': vaEstimates}
    
#------------------------------------------------------------------------------

def get_effectiveVaVc(evoSimFile):
    # get_effectiveVaVc() calculates estimates of va and vc from the adaptive 
    # event log files.
    
    # load evoSim object to ge tthe mcModel
    evoSim = get_evoSnapshot(evoSimFile)   
    
    # load abs log file and calculate the va estiamtes
    absLogFile  = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.get_adaptiveEventsLogFilename('abs'))[1] )
    dataAbs     = pd.read_csv(absLogFile)
    vaSimEst    = get_estimateRateOfAdaptFromSim(dataAbs,'abs',evoSim.mcModel)
    
    # load rel log file and calculate the vc estimates
    relLogFile  = os.path.join( os.path.split(evoSimFile)[0], os.path.split(evoSim.get_adaptiveEventsLogFilename('rel'))[1] )
    dataRel     = pd.read_csv(relLogFile)
    vcSimEst    = get_estimateRateOfAdaptFromSim(dataRel,'rel',evoSim.mcModel)
    
    # save the estimates to a dictionary for plotting. note that the data
    # vaSimEst and vcSimEst will be lists with the state space in the first
    # entry and the estimates in the second. state spaces that have not data
    # will have zero as the estimate.
    simVaVcEst = dict({'vaEst': vaSimEst, 'vcEst': vcSimEst})
    
    return simVaVcEst

# --------------------------------------------------------------------------

def get_evoSnapshot(evoSimSnapShotFile):
    # simple method to load the Sim data
    
    # save the data to a pickle file
    with open(evoSimSnapShotFile, 'rb') as file:
        # Serialize and write the variable to the file
        evoSim = pickle.load(file)
            
    return evoSim

# --------------------------------------------------------------------------

def save_evoSnapshot(evoSim):
    # save evoSim object to location of outputs
    
    with open(evoSim.get_evoSimFilename(), 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(evoSim, file)
    
    return None

# --------------------------------------------------------------------------

def get_mcModelFromEvoSim(evoSimSnapShotFile):
    # save evoSim object to location of outputs
    
    # save the data to a pickle file
    with open(evoSimSnapShotFile, 'rb') as file:
        # Serialize and write the variable to the file
        evoSim = pickle.load(file)
            
    return evoSim.mcModel

# --------------------------------------------------------------------------

def get_stateDataForHist(evoSimSnapShotFile):
    """
    Returns states and weights indicating time spent in each state. 
    inputs:
        evoSimSnapShotFile - pickle file with evo simulation snapshot
    outputs:
        idxGrp - states with nonzero counts
        idxCnt - count of iterations spent in each state to use as histogram weights
    """
    # 
    
    # save the data to a pickle file
    with open(evoSimSnapShotFile, 'rb') as file:
        # Serialize and write the variable to the file
        evoSim = pickle.load(file)
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # due to memory limitations we cannot use the data directly into histogram
    # plots. Instead we count and used the total count as weights.
    idxInt = np.floor(data['mean_b_idx'].values)
    idxGrp = np.sort(np.unique(idxInt))
    idxCnt = np.zeros(idxGrp.shape)
    
    for ii in range(len(idxGrp)):
        idxCnt[ii] = np.sum(np.sign(idxInt[idxInt==idxGrp[ii]]))
    
    return {'states': idxGrp,'wghts': idxCnt}

#%% ------------------------------------------------------------------------
############################################################################
#                           Plotting Functions   
############################################################################
# --------------------------------------------------------------------------

def plot_2dWave(nij,bm,cm):
    # simple function generate plots of the 2d wave
    
    nij_b = np.sum(nij,1)
    nij_c = np.sum(nij,0)
    
    nb = nij.shape[0]
    nc = nij.shape[1]
    
    bstates = [str(bm[ii,0]) for ii in range(nb)]
    cstates = [str(cm[0,ii]) for ii in range(nc)]
    
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=[12,5])
    im = ax1.imshow(np.log10(nij))
    
    # Add the colorbar, associating it with the image
    cbar = fig.colorbar(im, ax=ax1)
    
    # Set labels 
    ax1.set_xlabel('c-fitness state')
    ax1.set_ylabel('b-fitness state')
    cbar.set_label(r'$\log_10$ of abundances')
    
    ax1.set_xticks([ii for ii in range(nc)])
    ax1.set_yticks([ii for ii in range(nb)])
    ax1.set_xticklabels(cstates)
    ax1.set_yticklabels(bstates)
    
    ax2.bar(bstates,nij_b)
    ax2.set_xlabel("b-fitness state")
    ax2.set_ylabel(r'$\log_10$ of abundances')
        
    ax3.bar(cstates,nij_c)
    ax3.set_xlabel("c-state")
    ax3.set_ylabel(r'$\log_10$ of abundances')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

    return None

# --------------------------------------------------------------------------

def plot_selection_coeff(outputfile,fitType):
    # plots the selection coeff from simulation versus theory
    # assumes d-terms are constant 
    
    # load selection dynamics file
    data = pd.read_csv(outputfile)
    
    # get d-term
    dval = data['d_term'][0]
    tau  = 1.0/(dval-1.0)
    
    # get estimate of the selection coefficient per iteration
    sEst = tau * np.log(data['p01'][1:].values/data['p01'][0:-1].values)
    if (fitType=='abs'):
        sThr = data['sa'][0:-1].values
        s_str = 'sa_'
    else:
        sThr = data['sc'][0:-1].values
        s_str = 'sc_'
    
    # get times
    tt = data['t'][1:].values/tau
    ttt = data['t'].values/tau
    
    # get frequencies    
    pSim = data['p01'].values
    pThr = data['p01'][0]*np.exp(sThr[0]*ttt)
    
    # generate the figures
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=[12,6])
    ax1.scatter(tt,sEst,c='blue',label=s_str+'Sim')
    ax1.plot(tt,sThr,c='red',label=s_str+'Thr')
    ax1.set_xlabel('time (generations)')
    ax1.set_ylabel('selection coeff.')
    ax1.legend()
    
    ax2.plot(ttt,pSim,c='blue',label='pSim')
    ax2.plot(ttt,pThr,c='red',label='pThr')
    ax2.legend()
    
    plt.tight_layout()
    
    # save figure
    save_fig_name = outputfile.replace('.csv','selcoeff.png')
    plt.savefig(save_fig_name)
    
    return None

# --------------------------------------------------------------------------

def plot_rateOfAdaptation_Abs(outputfile):
    # estimates the rate of adaptation in absolute fitness. v for relative
    # cannot be calculated due to renormalization of c-terms.
    
    # load selection dynamics file
    data = pd.read_csv(outputfile)
    
    # get the required data
    time = data['time'].values
    biMax = data['max_bi'].values
    dTerm = data['d_term'][0]
    tau = 1/(dTerm-1)
    
    vaThr = data['va_imax'][1:].values

    nest = biMax.shape[0]-1
    
    vaEst = np.zeros([nest,1])
    tt = time[0:-1]
    
    for ii in range(nest):
        sGen  = lmfun.get_b_SelectionCoeff(biMax[ii+1],biMax[ii],dTerm)
        tGen = (time[ii+1]-time[ii])/tau
        vaEst[ii] = sGen/tGen


    # generate the figures
    fig,ax = plt.subplots(1,1,figsize=[6,6])
    ax.scatter(tt,vaEst,c='blue',label='vb_Sim')
    ax.plot(tt,vaThr,c='red',label='vb_Thr')
    ax.set_xlabel('time (iteration)')
    ax.set_ylabel('Rate of Adaptation (fit/gen)')
    ax.legend()
    
    plt.tight_layout()
    
    return None

# --------------------------------------------------------------------------

def plot_evoMcModel(mcModel):
    # plot the MC model 
    
    fig,ax = plt.subplots(1,1,figsize=[5,5])
    ax.plot(mcModel.state_i, mcModel.va_i,c='blue',label='vb')
    ax.plot(mcModel.state_i, mcModel.vc_i,c='red',label='vc')
    ax.plot(mcModel.state_i, mcModel.ve_i,c='black',label='vE')
    ax.set_xlabel('absolute fitness state')
    ax.set_ylabel('rate of adaptation')
    ax.legend()
    
    return None

# --------------------------------------------------------------------------

def plot_simulationAnalysis(evoSim):
    # Plot key figures to analyze the dynamics of the travelling b/c fitness wave
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # Figures to plot include the mean fitnes abs and relative
    # 1. MC model
    # 2. Mean fitness over time
    # 3. b-idx width over time
    # 4. c-idx width over time
    
    # get the data
    time = data['time'].values
    tau  = 1/(data['d_term'][0]-1)
    tt   = time/tau
    
    yavg = data['gamma'].values
    
    bidx_min = data['min_b_idx'].values
    bidx_max = data['max_b_idx'].values
    bidx_avg = data['mean_b_idx'].values
    bidx_mod = data['mode_b_idx'].values
    bidx_qi  = data['mean_b_idx'].values + data['qi_bavg'].values
    
    cidx_min = data['min_c_idx'].values
    cidx_max = data['max_c_idx'].values
    cidx_avg = data['mean_c_idx'].values
    cidx_qi  = data['mean_c_idx'].values + data['qi_cavg'].values
    
    idxss = evoSim.mcModel.get_mc_stable_state_idx()-1
    idxav = np.mean(bidx_avg)
    vdx = evoSim.mcModel.va_i[idxss]
    
    fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=[12,12])
    
    # MC model
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax11.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax11.set_xlabel('absolute fitness state')
    ax11.set_ylabel('rate of adaptation')
    ax11.legend()
    ax11t = ax11.twinx()
    ax11t.hist(bidx_avg,alpha = 0.5, color= 'k')
    ax11t.set_yticks([])
    
    idxlbl = ("i_ss=%d, i_av=%d, T1E%d, se=%.0E" % (idxss,idxav,int(np.log10(evoSim.params['T'])),evoSim.params['se']))
    ax11.text(idxss-15,0*vdx,idxlbl, fontsize = 12)             
    
    # secax11 = ax11.secondary_yaxis('right')
    # secax11.histogram(bidx_avg)
    
    # Mean gamma over time (generations)
    ax12.plot(tt , yavg ,c='black',label='avg density')
    ax12.set_xlabel('time (gen)')
    ax12.set_ylabel('pop density')
    ax12.legend()
    
    # b-fitness width
    ax21.plot(tt, bidx_min, c='blue', label='i_min_b')
    ax21.plot(tt, bidx_max, c='green', label='i_max_b')
    ax21.plot(tt, bidx_avg, c='red', label='i_avg_b')
    ax21.plot(tt, bidx_mod, c='magenta', label='i_mod_b')
    ax21.plot(tt, bidx_qi, c='black', label='i_qi_b')
    ax21.set_xlabel('time (generations)')
    ax21.set_ylabel('b-state')
    ax21.legend()
    
    # c-fitness width
    ax22.plot(tt, cidx_min, c='blue', label='i_min_c')
    ax22.plot(tt, cidx_max, c='green', label='i_max_c')
    ax22.plot(tt, cidx_avg, c='red', label='i_avg_c')
    ax22.plot(tt, cidx_qi, c='black', label='i_qi_c')
    ax22.set_xlabel('time (generations)')
    ax22.set_ylabel('c-state')
    ax22.legend()
    
    # save figure in location where outputs are located
    figName = evoSim.outputStatsFile.replace('.csv','.png')
    fig.savefig(figName,bbox_inches='tight')
    return None

# --------------------------------------------------------------------------

def plot_simulationAnalysis_v2(evoSim):
    # Plot key figures to analyze the dynamics of the travelling b/c fitness wave
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # Figures to plot include the mean fitnes abs and relative
    # 1. MC model
    # 2. Mean fitness over time
    # 3. b-idx width over time
    # 4. c-idx width over time
    
    # get the data
    time = data['time'].values
    tau  = 1/(data['d_term'][0]-1)
    tt   = time/tau
    
    yavg = data['gamma'].values
    
    bidx_min = data['min_b_idx'].values
    bidx_max = data['max_b_idx'].values
    bidx_avg = data['mean_b_idx'].values
    bidx_mod = data['mode_b_idx'].values
    bidx_qi  = data['mean_b_idx'].values + data['qi_bavg'].values
    
    cidx_min = data['min_c_idx'].values
    cidx_max = data['max_c_idx'].values
    cidx_avg = data['mean_c_idx'].values
    cidx_qi  = data['mean_c_idx'].values + data['qi_cavg'].values
    
    idxss = evoSim.mcModel.get_mc_stable_state_idx()-1
    idxav = np.mean(bidx_avg)
    vdx = evoSim.mcModel.va_i[idxss]
    
    fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=[12,12])
    
    # MC model
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax11.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax11.set_xlabel('absolute fitness state')
    ax11.set_ylabel('rate of adaptation')
    ax11.legend()
    ax11t = ax11.twinx()
    ax11t.boxplot(bidx_avg,vert=False)

    idxlbl = ("i_ss=%d, i_av=%d, T1E%d, se=%.0E" % (idxss,idxav,int(np.log10(evoSim.params['T'])),evoSim.params['se']))
    ax11.text(idxss+10,0*vdx,idxlbl, fontsize = 12)       
    
    # secax11 = ax11.secondary_yaxis('right')
    # secax11.histogram(bidx_avg)
    
    # Mean gamma over time (generations)
    ax12.plot(tt , yavg ,c='black',label='avg density')
    ax12.set_xlabel('time (gen)')
    ax12.set_ylabel('pop density')
    ax12.legend()
    
    # b-fitness width
    ax21.plot(tt, bidx_min, c='blue', label='i_min_b')
    ax21.plot(tt, bidx_max, c='green', label='i_max_b')
    ax21.plot(tt, bidx_avg, c='red', label='i_avg_b')
    ax21.plot(tt, bidx_mod, c='magenta', label='i_mod_b')
    ax21.plot(tt, bidx_qi, c='black', label='i_qi_b')
    ax21.set_xlabel('time (generations)')
    ax21.set_ylabel('b-state')
    ax21.legend()
    
    # c-fitness width
    ax22.plot(tt, cidx_min, c='blue', label='i_min_c')
    ax22.plot(tt, cidx_max, c='green', label='i_max_c')
    ax22.plot(tt, cidx_avg, c='red', label='i_avg_c')
    ax22.plot(tt, cidx_qi, c='black', label='i_qi_c')
    ax22.set_xlabel('time (generations)')
    ax22.set_ylabel('c-state')
    ax22.legend()
    
    return None

# --------------------------------------------------------------------------

def plot_simulationAnalysis_RateEstimate(evoSim):
    # Plot key figures to analyze the dynamics of the travelling b/c fitness wave
    # this function should only be used with runs of the simulation that have set
    # the rate of environmental chance to zero.
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # Figures to plot include the mean fitnes abs and relative
    # 1. MC model
    # 2. Mean fitness over time
    # 3. b-idx width over time
    # 4. c-idx width over time
    
    # get the data
    time = data['time'].values
    tau  = 1/(data['d_term'][0]-1)
    tt   = time/tau
    
    yavg = data['gamma'].values
    
    bidx_min = data['min_b_idx'].values
    bidx_max = data['max_b_idx'].values
    bidx_avg = data['mean_b_idx'].values
    bidx_mod = data['mode_b_idx'].values
    bidx_qi  = data['mean_b_idx'].values + data['qi_bavg'].values
    
    cidx_min = data['min_c_idx'].values
    cidx_max = data['max_c_idx'].values
    cidx_avg = data['mean_c_idx'].values
    cidx_qi  = data['mean_c_idx'].values + data['qi_cavg'].values
    
    idxss = evoSim.mcModel.get_mc_stable_state_idx()-1
    idxav = np.mean(bidx_avg)
    vdx = evoSim.mcModel.va_i[idxss]
    
    # va-estimates
    bidx_avg = np.floor(data['mean_b_idx'].values)
    init_state = np.max(evoSim.simInit.bij_mutCnt)
    max_state = np.max(evoSim.mcModel.state_i)
    
    nCycles = np.sum(np.abs(np.diff(bidx_avg))>10) + (bidx_avg[-1] - init_state)/(max_state-init_state)
    
    va_idx = np.unique(bidx_avg)
    va_est = np.zeros(va_idx.shape)
    
    for ii in range(len(va_idx)):
        # va_est[ii] = 1.6E-5 / (np.sum(bidx_avg == (va_idx[ii]))/500000) # 
        # va_est[ii] = 0.033 * evoSim.mcModel.sa_i[int(va_idx[ii])] / (tau * np.sum(bidx_avg == (va_idx[ii]))* nCycles / 500000)
        Tp = ( 500000 / nCycles )
        fp = ( np.sum(bidx_avg == (va_idx[ii])) / 500000 )
        Ts = fp*Tp/tau
        va_est[ii] =  evoSim.mcModel.sa_i[int(va_idx[ii]+1)] / Ts
                                       
    fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=[12,12])
    
    # MC model
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax11.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax11.set_xlabel('absolute fitness state')
    ax11.set_ylabel('rate of adaptation')
    ax11.legend()
    ax11t = ax11.twinx()
    ax11t.hist(bidx_avg,alpha = 0.5, color= 'k')
    ax11t.set_yticks([])
    
    idxlbl = ("i_ss=%d, i_av=%d, T1E%d, se=%.0E" % (idxss,idxav,int(np.log10(evoSim.params['T'])),evoSim.params['se']))
    ax11.text(idxss-15,0*vdx,idxlbl, fontsize = 12)             
    
    # secax11 = ax11.secondary_yaxis('right')
    # secax11.histogram(bidx_avg)
    
    # MC model
    ax12.scatter(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax12.scatter(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax12.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax12.scatter(va_idx,va_est,c='cyan',label='vb_est')
    
    ax12.set_xlabel('absolute fitness state')
    ax12.set_ylabel('rate of adaptation')
    ax12.legend()
    
    # b-fitness width
    ax21.plot(tt, bidx_min, c='blue', label='i_min_b')
    ax21.plot(tt, bidx_max, c='green', label='i_max_b')
    ax21.plot(tt, bidx_avg, c='red', label='i_avg_b')
    ax21.plot(tt, bidx_mod, c='magenta', label='i_mod_b')
    ax21.plot(tt, bidx_qi, c='black', label='i_qi_b')
    ax21.set_xlabel('time (generations)')
    ax21.set_ylabel('b-state')
    ax21.legend()
    
    # c-fitness width
    ax22.plot(tt, cidx_min, c='blue', label='i_min_c')
    ax22.plot(tt, cidx_max, c='green', label='i_max_c')
    ax22.plot(tt, cidx_avg, c='red', label='i_avg_c')
    ax22.plot(tt, cidx_qi, c='black', label='i_qi_c')
    ax22.set_xlabel('time (generations)')
    ax22.set_ylabel('c-state')
    ax22.legend()
    
    return None

# --------------------------------------------------------------------------

def plot_simulationAnalysis_comparison(evoSim1,evoSim2):
    # Plot key figures to analyze the dynamics of the travelling b/c fitness wave
    
    # load selection dynamics file
    data1 = pd.read_csv(evoSim1.outputStatsFile)
    data2 = pd.read_csv(evoSim2.outputStatsFile)
    
    # get the data
    bidx_avg1 = data1['mean_b_idx'].values
    bidx_avg2 = data2['mean_b_idx'].values
    bidx_avg = [bidx_avg1,bidx_avg2]
    
    lb1 = "1E%d" % (int(np.log10(evoSim1.mcModel.params['T'])))
    lb2 = "1E%d" % (int(np.log10(evoSim2.mcModel.params['T'])))
    
    fig,ax = plt.subplots(1,1,figsize=[12,12])
    
    # MC model
    ax.scatter(evoSim1.mcModel.state_i, evoSim1.mcModel.va_i,c='blue',label='vb_T'+lb1)
    ax.scatter(evoSim1.mcModel.state_i, evoSim1.mcModel.vc_i,c='red',label='vc_T'+lb1)
    ax.plot(evoSim1.mcModel.state_i, evoSim1.mcModel.ve_i,c='black',label='vE_T'+lb1)
    ax.scatter(evoSim2.mcModel.state_i, evoSim2.mcModel.va_i,c='blue',label='vb_T'+lb2,marker='.')
    ax.scatter(evoSim2.mcModel.state_i, evoSim2.mcModel.vc_i,c='red',label='vc_T'+lb2,marker='.')
    ax.plot(evoSim2.mcModel.state_i, evoSim2.mcModel.ve_i,c='grey',label='vE_T'+lb2,linestyle=':')
    ax.set_xlabel('absolute fitness state')
    ax.set_ylabel('rate of adaptation')
    ax.legend()
    axt = ax.twinx()
    axt.boxplot(bidx_avg,vert=False)
    
    return None

# --------------------------------------------------------------------------

def plot_environmentalChange(evoSim):
    # Plot fitness over time and estimates of the rate of environmental change
    # these plots were mainly developed to check that the rate of environmental
    # change was implemented adequately to achieve the set rate in the parameters
    #
    # key assumptions for running this function are
    #
    # Assumptions: 
    # 1. population is homogeneous in b-mutations (no b-evolution)
    # 2. environmental changes occur at non-trivial rate, with se > 0 decrease
    #    in fitness
    # 3. no d-evolution
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # Figures to plot include the mean fitnes abs and relative
    # 1. plot b-fitness over time (iterations)
    # 2. vE realizations historgram
    # 3. vE realizations across time (generations)
    # 4. vE realization across gamma
    
    # get the data
    time    = data['time'].values
    max_bi  = data['max_bi'].values
    dterm   = data['d_term'][0]    
    y       = data['gamma'].values
    envJmp  = data['envShft'].values
    tau     = 1.0/(dterm-1.0)
    tt      = time/tau 
    
    # calculate vE estimates, and get the indices where jmps occur
    output = get_rateOfEnvironmentalChange_samples(time,max_bi,dterm,envJmp)
    vE  = output[0]
    idxE= output[1]
    ttE = tt[idxE]
    yE  = y[idxE]
    
    # theoretical vE 
    vEThry = -evoSim.params['se']*evoSim.params['R']*tau*np.ones(vE.shape)
    
    fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=[12,12])
    
    # b-fitness vs generations
    ax11.plot(tt, max_bi,c='blue')
    ax11.set_xlabel('time (generations)')
    ax11.set_ylabel('population b-term')
    
    # vE estimates histogram (rel err)
    ax12.hist(np.log10(vE/vEThry))
    ax12.set_xlabel('log10(vE/vEthy)')
    
    # vE estimates vs generations
    ax21.scatter(ttE, np.log10(-vE), c='blue',label='vE_est')
    ax21.plot(ttE, np.log10(-vEThry), c='red',label='vE_thy')
    ax21.set_xlabel('time (generations)')
    ax21.set_ylabel('log10(vE)')
    ax21.legend()
    
    # vE estimates vs population density
    ax22.scatter(yE, np.log10(-vE), c='blue',label='vE_est')
    ax22.plot(yE, np.log10(-vEThry), c='red',label='vE_thy')
    ax22.set_xlabel('population density')
    ax22.set_ylabel('log10(vE)')
    ax22.legend()
    
    return None
#------------------------------------------------------------------------------

def plot_evoMcModel_withVaEst(evoSim,vaEst):
    # plot the MC model 
    
    fig,ax = plt.subplots(1,1,figsize=[5,5])
    ax.plot(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax.plot(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax.scatter(evoSim.mcModel.state_i[vaEst!=0],vaEst[vaEst!=0],facecolors='none', edgecolors='b',label='vbEst')
    ax.set_xlabel('absolute fitness state')
    ax.set_ylabel('rate of adaptation')
    ax.legend()
    
    return None

# --------------------------------------------------------------------------

def plot_mcModel_histAndVaEstimates(evoSim):
    # plot_vaEstimateMcChainPlots() takes the evo file and generates plots of 
    # of the MC state space with theoretical rates of adaptation, estimates of
    # the rates of adaptation, and rates of environmental change.
    
    # load selection dynamics file
    data = pd.read_csv(evoSim.outputStatsFile)
    
    # Figures to plot include the mean fitnes abs and relative
    # 1. MC model w/ histogram
    # 2. MC model w/ va estimates
    
    # get the data
    bidx_avg = data['mean_b_idx'].values

    idxss = evoSim.mcModel.get_mc_stable_state_idx()-1
    idxav = np.mean(bidx_avg)
    vdx = evoSim.mcModel.va_i[idxss]
    
    vaEst = calculate_RateOfAdapt_estimates(evoSim)
    
    fig,(ax11,ax12) = plt.subplots(1,2,figsize=[12,6])
    
    # MC model
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax11.scatter(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax11.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax11.set_xlabel('absolute fitness state')
    ax11.set_ylabel('rate of adaptation')
    ax11.legend()
    ax11t = ax11.twinx()
    ax11t.hist(bidx_avg,alpha = 0.5, color= 'k')
    ax11t.set_yticks([])
    
    idxlbl = ("i_ss=%d, i_av=%d, T1E%d, se=%.0E" % (idxss,idxav,int(np.log10(evoSim.params['T'])),evoSim.params['se']))
    ax11.text(idxss-15,0*vdx,idxlbl, fontsize = 12)             
    
    # Mean gamma over time (generations)
    ax12.plot(evoSim.mcModel.state_i, evoSim.mcModel.va_i,c='blue',label='vb')
    ax12.plot(evoSim.mcModel.state_i, evoSim.mcModel.vc_i,c='red',label='vc')
    ax12.plot(evoSim.mcModel.state_i, evoSim.mcModel.ve_i,c='black',label='vE')
    ax12.scatter(evoSim.mcModel.state_i[vaEst!=0],vaEst[vaEst!=0],facecolors='none', edgecolors='b',label='vbEst')
    ax12.set_xlabel('absolute fitness state')
    ax12.set_ylabel('rate of adaptation')
    ax12.legend()
    
    # save figure in location where outputs are located
    figName = evoSim.outputStatsFile.replace('.csv','_mcModel.png')
    fig.savefig(figName,bbox_inches='tight')
    
    return None

# --------------------------------------------------------------------------

def plot_mcModel_fromInputFile(paramfile,modelType,absFitType):
    # plot_mcModel_fromInputFile() generates the MC model from a parameter file
    
    # load selection dynamics file
    
    mcEvoOptions    = evoObj.evoOptions(paramfile,modelType,absFitType)
    mcModel         = mcFac.mcFactory().createMcModel( mcEvoOptions )
    
    # Figures to plot include the mean fitnes abs and relative
    idxss = int(mcModel.get_mc_stable_state_idx()-1)
    vdx   = mcModel.va_i[idxss]
    rho   = mcModel.calculate_evoRho()
    
    
    
    fig,ax = plt.subplots(1,1,figsize=[6,6])
    
    # MC model
    ax.scatter(mcModel.state_i, mcModel.va_i,c='blue',label='vb')
    ax.scatter(mcModel.state_i, mcModel.vc_i,c='red',label='vc')
    
    ax.plot(mcModel.state_i, mcModel.ve_i,c='black',label='vE')
    ax.set_xlabel('absolute fitness state')
    ax.set_ylabel('rate of adaptation')
    ax.legend()
    
    idxlbl = ("i_ss=%d, rho=%.2f" % (idxss,rho))
    ax.text(15,0*vdx,idxlbl, fontsize = 12)                 
    
    return None

# --------------------------------------------------------------------------
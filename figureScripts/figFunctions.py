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

from evoLibraries.LotteryModel import LM_functions as lmfun

#%% ------------------------------------------------------------------------
#                               Functions   
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
#                           Plotting Functions   
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
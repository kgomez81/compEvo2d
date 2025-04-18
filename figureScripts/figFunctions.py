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
    tau  = 1/(dval-1)
    
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
    
    fig,ax = plt.subplots(1,1,figsize=[5,5])
    ax.plot(mcModel.state_i, mcModel.va_i,c='blue')
    ax.plot(mcModel.state_i, mcModel.vc_i,c='red')
    ax.plot(mcModel.state_i, mcModel.ve_i,c='black')
    ax.set_xlabel('state')
    ax.set_ylabel('rate of adaptation')
    
    return None

# --------------------------------------------------------------------------

def plot_simulationAnalysis(outputfile):
    # plot the travelling waves by state space indices
    # also include the 
    
    # load selection dynamics file
    data = pd.read_csv(outputfile)
    
    return None
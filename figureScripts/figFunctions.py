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
    
    return None
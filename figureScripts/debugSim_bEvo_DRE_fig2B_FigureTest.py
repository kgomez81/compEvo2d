# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:36:11 2025

@author: Owner
"""

import numpy as np
import pickle

import os
import sys
sys.path.insert(0, os.getcwd() + '\\..')

from evoLibraries.SimRoutines import SIM_Init as evoInit
from evoLibraries.SimRoutines import SIM_DRE_class as simDre

import figFunctions as figfun


def get_evoSnapshot(evoSimSnapShotFile):
        
        # save the data to a pickle file
        with open(evoSimSnapShotFile, 'rb') as file:
            # Serialize and write the variable to the file
            evoSim = pickle.load(file)
                
        return evoSim

def main():
    
    # set the file with evoSim saved
    # env low both
    case = 1
    workdir = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B_FINAL\\'
    
    if case == 0:
        # env high both se = 0.05
        simFile1 = 'sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250516_135316_evoSim'
        simFile2 = 'sim_Fig2B_T1E12_snpsht_param_03B_DRE_bEvo_20250516_134358_evoSim'
    elif case == 1:
        # env low both se = 0.03
        simFile1 = 'sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250516_111306_evoSim'
        simFile2 = 'sim_Fig2B_T1E12_snpsht_param_03B_DRE_bEvo_20250516_131129_evoSim'
    elif case == 2:
        # env high both se = 0.05 (long runs)
        simFile1 = 'sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250516_144609_evoSim'
        simFile2 = 'sim_Fig2B_T1E12_snpsht_param_03B_DRE_bEvo_20250516_145041_evoSim'
    elif case == 3:
        # env high both se = 0.05 (long runs)
        # evo with/without interference
        simFile1 = 'sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250516_144609_evoSim'
        simFile2 = 'sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250516_152103_evoSim'
        
    #simFile2 = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B_FINAL\\sim_Fig2B_T1E12_snpsht_param_03B_DRE_bEvo_20250516_132701_evoSim.pickle'
    
    # load the evoSim object
    evoSim1 = get_evoSnapshot(workdir+'\\'+simFile1+'.pickle')
    evoSim2 = get_evoSnapshot(workdir+'\\'+simFile2+'.pickle')
    
    # plot the evoSim snapshot with data
    figfun.plot_simulationAnalysis(evoSim1)
    figfun.plot_simulationAnalysis(evoSim2)

    figfun.plot_simulationAnalysis_comparison(evoSim1, evoSim2)
    
    return None


if __name__ == "__main__":

    main()

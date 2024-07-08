# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:08:31 2022
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np

import evoLibraries.SimRoutines.SIM_class as sim

class simDREClass(sim.simClass):
    # Class for simulating evolution with DRE type absolute fitness state space 
    #
    # Members of simClass 
    # self.params     = mcEvoOptions.params         # dictionary with evo parameters
    # self.absFitType = mcEvoOptions.absFitType     # absolute fitness evolution term
    
    # # 2d array data
    # self.nij        = np.zeros([1,1])       # array for genotype abundances
    # self.bij        = np.zeros([1,1])       # array for genotype b-terms 
    # self.dij        = np.zeros([1,1])       # array for genotype d-terms 
    # self.cij        = np.zeros([1,1])       # array for genotype c-terms 
    
    # # 1d array and other data
    # self.iAbsMutCnt = np.zeros([1,1])       # stores abs fit mutation counts along 2d arry rows
    # self.jRelMutCnt = np.zeros([1,1])       # stores rel fit mutation counts along 2d arry rows
    # self.tmax       = 10                    # max iterations
    
    # # - mcModel is associated Markov Chain used to reference state space
    # # - fitMapEnv maps fitnes decline from environmental changes 
    # self.mcModel    = []
    # self.fitMapEnv  = np.zeros([1,1])    
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,mcEvoOptions):
        
        # call constructor of base class
        super().__init__(mcEvoOptions)
        
    #%% ------------------------------------------------------------------------
    # Definitions for abstract methods
    # --------------------------------------------------------------------------
    
    def init_evolutionModel(self):
        # Method that generates the initial values of the evolution model using
        # the associated Markov Chain model                                    
        
        return None
    
    # --------------------------------------------------------------------------
    
    def get_populationMutations(self):
        # Method that defines mutations of genotypes. Mutations depend on the  
        # the type of absolute fitness evoltuion model selected. This method   
        # is best applied to the array form of abundances.                     
        #                                                                     
        # Absolute fitness mutations add/modify rows of the 2d evolution arrays
        # while relative fitness mutations add columns to 2d evo arrays        
        
        return None

    #%% ------------------------------------------------------------------------
    # List of conrete methods from MC class
    # --------------------------------------------------------------------------
    
    
    #%% ----------------------------------------------------------------------------
    #  Specific methods for the DRE MC class
    # ------------------------------------------------------------------------------
    
    
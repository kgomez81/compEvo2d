# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:09:10 2022

@author: Kevin Gomez

This file defines the simualtion class, which is used to run full simulations of 
evoluation with selection defined by the variable density lottery model

Note: This class requires two libraries: LM_functions and SIM_Functions

"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np

from abc import ABC, abstractmethod
from joblib import Parallel, delayed, cpu_count

import evoLibraries.LotteryModel.LM_functions as lmFun
import SIM_functions as simfun

class simClass(ABC):
    # ABSTRACT class used to carry out evolution simulations
    #
    # IMPORTANT: This class is not mean to be used for anything other then for 
    # consolidating the common code of RM and DRE SIM classes, i.e. something like 
    # an Abstract class
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,mcEvoOptions,evoInit):
        # evoInit is dictionary with initial values for evor arrays below.
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        self.params     = mcEvoOptions.params         # dictionary with evo parameters
        self.absFitType = mcEvoOptions.absFitType     # absolute fitness evolution term
        self.tmax       = 10                    # max iterations (default is 10)
        
        # 2d array with adult abundances
        self.nij        = np.zeros([1,1])       # array for genotype abundances (adults only)
        
        # 2d arrays which store mutation counts
        self.bij_mutCnt     = np.zeros([1,1])   # array for genotype b-terms mutation counts
        self.dij_mutCnt     = np.zeros([1,1])   # array for genotype d-terms mutation counts
        self.cij_mutCnt     = np.zeros([1,1])   # array for genotype c-terms mutation counts
        
        self.mut_ij     = np.zeros([1,1])       # array for genotype mutant juventiles
        self.stoch_ij   = []                    # list of stochastic classes     
        self.stochThrsh = 100                   # threshold to assign with stochastic behavior
        
        # - mcModel is associated Markov Chain used to reference state space
        # - fitMapEnv maps fitnes decline from environmental changes 
        self.mcModel    = []
        self.fitMapEnv  = np.zeros([1,1])   
        
        # initialize arrays
        self.init_evolutionModel(evoInit)
        
    #%% ------------------------------------------------------------------------
    # Abstract methods
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def init_evolutionModel(self):
        "Method that generates the initial values of the evolution model using"
        "the associated Markov Chain model                                    "
        pass

    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_populationMutations(self):
        "Method that defines mutations of genotypes. Mutations depend on the  "
        "the type of absolute fitness evoltuion model selected. This method   "
        "is best applied to the array form of abundances.                     "
        "                                                                     "
        "Absolute fitness mutations add/modify rows of the 2d evolution arrays"
        "while relative fitness mutations add columns to 2d evo arrays        "
        pass

    # --------------------------------------------------------------------------
    
    @abstractmethod
    def run_environmentalDegredation(self):
        "Method that defines how environmental degredation runs. This will    "
        "depend on the model and type of evolution.                           "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_bij(self):
        "Method to calculate array of bij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_dij(self):
        "Method to calculate array of dij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_cij(self):
        "Method to calculate array of cij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        pass
    
    #%% ------------------------------------------------------------------------
    # Class methods
    # --------------------------------------------------------------------------
    
    def run_evolutionModel(self):
        # main function to run evolutionary model
        
        # set time index
        t           = 0
        popExtinct  = False 
        
        # run model tmax generations or until population is extinct
        while ( (t < self.tmax) and not (self.popSize() < 10) ):
            
            # get new juveniles and modify arrays for potentially new classes
            self.get_populationMutations()
            
            # run competitino phase of model
            self.run_competitionPhase()
            
            # run death phase of model
            self.run_deathPhase()
            
            # update time increment
            t = t + 1
        
        return None
        
    #------------------------------------------------------------------------------
    
    def run_competitionPhase(self):
        # run_competitionPhase() generates the set of juvenilees and 
        # simulates competition to determine the number of adults prior
        # to the death phase.
        
        # NOTE: get_populationMutations() should expand the arrays to capture 
        # the appearance of juvenile mutants. How this occurs is unique to the 
        # type of model (d vs b evo, RM vs DRE)
        # 
        
        
        # COMPETITION for DETERMINISTIC CLASSES
        # first we calculate everything as deterministic, then we recalculate
        # outcomes from stochastic competitions 
        
        
        # COMPETITION for STOCHASTIC CLASSES
        
    
        return None
    #------------------------------------------------------------------------------
    
    def run_deathPhase(self):
        # run_deathPhase() gernates the set of new and remaining adults 
        # following death.
        
        # DEATH for STOCHASTIC CLASSES
        
        # DEATH for DETERMINISTIC CLASSES
        
        # apply death phase to stochastically modeled classes
        for ii in range(self.nij.shape[0]):
            for jj in range(self.nij.shape[1]):
                # sample deaths using binomial distribution
                self.nij[ii,jj] = np.random.binomial(self.nij[ii,jj],1/self.dij[ii,jj],1)
        
        return None

    #------------------------------------------------------------------------------
    
    def get_stochasticFitnessClasses(self):
        # get_stochasticFitnessClasses() identifies the set of genotypes that are 
        # below the stochastic threshold, whose abundances will not be modeled 
        # deterministically.
        
        # clear current list
        self.stoch_ij = []
        
        # calculate the stochastic threshold
        for ii in range(self.nij.shape[0]):
            for jj in range(self.nij.shape[1]):
                if (self.nij[ii,jj] < self.stochThrsh) and (self.nij[ii,jj] > 0):
                    self.stoch_ij.append([ii,jj])
        
        return None
    
    #------------------------------------------------------------------------------
    
    def get_arrayMap(self):
        # Used to flattens 2d arrays so that they can be used by SIM_functions
        # the map allows calculated values from functions to be placed back 
        # to a 2d array
        
        ijMap = []
        
        for ii in range(self.nij.shape[0]):
            for jj in range(self.nij.shape[1]):
                if (self.nij[ii,jj] > 0):
                    ijMap.append([ii,jj])
        
        return ijMap
    
    #------------------------------------------------------------------------------
    
    def get_evoArraysExpand(self):
        # get_evoArraysExpand() expands evo arrays with additional rows/cols 
        # so that mutations can be added. Two key steps are:
        #   1. pads the set of array storing the evolutioanry state of the population
        #   2. fills in the padded the bij, dij and cij 
        #
        
        # expand adult abundances
        self.nij = np.pad(self.nij,1)
        
        # expand bij arrays
        self.bij = np.pad(self.bij,1)
        
        # expand dij arrays
        self.dij = np.pad(self.dij,1)
        
        # expand cij array
        self.cij = np.pad(self.cij,1)
        
        return None
    
    #------------------------------------------------------------------------------
    
    def get_evoArraysCollapse(self):
        # get_evoArraysCollapse() collapses evo arrays to remove any rows/cols
        # with only zero entries
        
        return None
    
    #------------------------------------------------------------------------------
    
    def get_bbar(self):
        # get_bbar() returns population mean value of the b-terms
        
        bbar = np.sum(self.nij*self.bij)/self.popSize()
        
        return bbar
    
    #------------------------------------------------------------------------------
    
    def get_dbar(self):
        # get_dbar() returns population mean value of the d-terms
        
        dbar = np.sum(self.nij*self.dij)/self.popSize()
        
        return dbar
    
    #------------------------------------------------------------------------------
    
    def get_cbar(self):
        # get_cbar() returns population mean value of the c-terms
        
        cbar = np.sum(self.nij*self.cij)/self.popSize()
        
        return cbar
    
    #------------------------------------------------------------------------------
    
    def get_U(self):    
        # get_U() returns the number of unoccupied territories
        
        U = self.params['T']-self.popSize()    
        
        return U
    
    #------------------------------------------------------------------------------
    
    def get_mij(self):
        # get_mij() returns the expected juveniles 
        
        mij = self.nij*self.bij*(self.get_U()/self.params['T'])
        
        return mij
    
    #------------------------------------------------------------------------------
    
    def get_lij(self):
        # get_lij() returns the lottery model lij parameters for each genotype, 
        # which are the poisson juvenile birth rates over unoccupied territories 
        
        lij = self.get_mij()/self.get_U()
        
        return lij
    
    #------------------------------------------------------------------------------
    
    def get_L(self):
        # get_L() returns the lottery model L parameter (sum of the poisson birth 
        # parameters over non-occupied territories)
        
        L = np.sum(self.get_lij())
        
        return L
    
    #------------------------------------------------------------------------------
    
    def store_evoSnapshot():
        
        # function to store evolutionary state
        # - time
        # - nij, bij, cij, dij
        
        # snapshot of population
        
        # snapshot of parameters (as csv file)
        
    
        return None
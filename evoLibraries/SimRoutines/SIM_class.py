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
    def __init__(self,mcEvoOptions):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        self.params     = mcEvoOptions.params         # dictionary with evo parameters
        self.absFitType = mcEvoOptions.absFitType     # absolute fitness evolution term
        
        # 2d array data
        self.nij        = np.zeros([1,1])       # array for genotype abundances
        self.bij        = np.zeros([1,1])       # array for genotype b-terms 
        self.dij        = np.zeros([1,1])       # array for genotype d-terms 
        self.cij        = np.zeros([1,1])       # array for genotype c-terms 
        self.tmax       = 10                    # max iterations
        
        # 1d array mappings to quickly calculate transitions
        # - we use the bi or di as a reference to map states i to bi/di-terms 
        # - fitMapEnv is a map for how to apply environmental degredation
        self.iState     = np.zeros([1,1])       # absolute fitness state space
        self.bi         = np.zeros([1,1])       # b-state space (bound by iMax)
        self.di         = np.zeros([1,1])       # d-state space (bound by iMax) 
        self.fitMapEnv  = np.zeros([1,1])       # map of fitness declines
        
    #%% ------------------------------------------------------------------------
    # Abstract methods
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
    
    #%% ------------------------------------------------------------------------
    # Class methods
    # --------------------------------------------------------------------------
    
    def init_evolutionModel(self,evoInit):
        # init_evolutionModel() is used to provide initial values for the population
        
        # 2d array data
        self.nij        = evoInit.nij      # array for genotype abundances
        self.bij        = evoInit.bij      # array for genotype b-terms 
        self.dij        = evoInit.dij      # array for genotype d-terms 
        self.cij        = evoInit.cij      # array for genotype c-terms 
        self.tmax       = evoInit.tmax     # max iterations
        
        # 1d array mappings to quickly calculate transitions
        # - we use the bi or di as a reference to map states i to bi/di-terms 
        # - fitMapEnv is a map for how to apply environmental degredation
        self.iState     = evoInit.iState        # absolute fitness state space
        self.bi         = evoInit.bi            # b-state space (bound by iMax)
        self.di         = evoInit.di            # d-state space (bound by iMax) 
        self.fitMapEnv  = evoInit.fitMapEnv     # map of fitness declines
        
        return None
    
    #------------------------------------------------------------------------------    
    
    def run_evolutionModel(self):
        # main function to run evolutionary model
        
        # set time index
        t           = 0
        popExtinct  = False 
        
        # run model tmax generations or until population is extinct
        while ( (t < self.tmax) and (not popExtinct) ):
            
            # run competitino phase of model
            self.run_competitionPhase()
            
            # run death phase of model
            self.run_deathPhase()
            
            # update time increment
            t = t + 1
            
            # check if population has become extinct
            if (self.popSize() < 10):
                popExtinct = True
        
        return None
    
    #------------------------------------------------------------------------------
    
    def run_competitionPhase(self):
        # run_competitionPhase() generates the set of juvenilees and 
        # simulates competition to determine the number of adults prior
        # to the death phase.
        

        
        # get the poisson parameters for juveniles produced lij = mij/U
        
        
        # get U array for competitions
        
        # run compeitions 
    
        return None
    #------------------------------------------------------------------------------
    
    def run_deathPhase(self):
        # run_deathPhase() gernates the set of new and remaining adults 
        # following death.
        
        # apply death phase to stochastically modeled classes
        for ii in range(self.dij.shape[0]):
            for jj in range(self.dij.shape[1]):
                # sample deaths using binomial distribution
                self.dij[ii,jj] = 0
        
        return None
    
    #------------------------------------------------------------------------------
    
    def run_environmentalDegredation():
        # run_environmentalDegredation() simulations a change in the environment that
        # reduces absolute fitness of all genotypes. Environmental degredation reduces
        # fitness by the same amount for all types.
        
        # for now, we will implement the simplest strategy of shifting back one state
        
        return None

    #------------------------------------------------------------------------------
    
    def get_stochasticFitnessClasses(self):
        # get_stochasticFitnessClasses() identifies the set of genotypes that are 
        # below the stochastic threshold, whose abundances will not be modeled 
        # deterministically.
        
        # calculate the stochastic threshold
        
        
        return ijStoch
    
    #------------------------------------------------------------------------------
    
    def get_arrayMap(self):
        # Used to flattens 2d arrays so that they can be used by SIM_functions
        # the map allows calculated values from functions to be placed back 
        # to a 2d array
        
        return ijMap
    
    #------------------------------------------------------------------------------
    
    def get_evoArraysExpand(self):
        
        return None
    
    #------------------------------------------------------------------------------
    
    def get_evoArraysCollapse(self):
        
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
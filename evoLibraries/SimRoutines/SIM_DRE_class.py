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
import evoLibraries.MarkovChain.MC_factory as mcFac

class simDREClass(sim.simClass):
    # Class for simulating evolution with DRE type absolute fitness state space 
    #
    # # Members of simClass 
    # self.params     = mcEvoOptions.params         # dictionary with evo parameters
    # self.absFitType = mcEvoOptions.absFitType     # absolute fitness evolution term
    # self.tmax       = 10                          # max iterations (default is 10)
    
    # # 2d array data
    # self.nij        = np.zeros([1,1])       # array for genotype abundances
    # self.mij        = np.zeros([1,1])       # array for genotype abundances
    # self.bij_mutCnt     = np.zeros([1,1])   # array for genotype b-terms mutation counts
    # self.dij_mutCnt     = np.zeros([1,1])   # array for genotype d-terms mutation counts
    # self.cij_mutCnt     = np.zeros([1,1])   # array for genotype c-terms mutation counts
    
    # # - mcModel is associated Markov Chain used to reference state space
    # self.mcModel    = []
    # # - fitMapEnv maps fitness declines from environmental change across the state space
    # #   to try and achieve homogenous fitness declines across the state space. 
    # self.fitMapEnv  = np.zeros([1,1])   
    
    # # Simulation flag - 
    # #   true: stochastic birth/comp/death simulated, 
    # #   false: poisson sampling with expectation given dens-dep lottery model Eqtns
    # self.fullStochModelFlag = False         
    
    # # initialize arrays
    # self.init_evolutionModel(evoInit)  
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,mcEvoOptions):
        
        # call constructor of base class
        super().__init__(mcEvoOptions)
        
    #%% ------------------------------------------------------------------------
    # Definitions for abstract methods
    # --------------------------------------------------------------------------
    
    def init_evolutionModel(self,evoInit):
        "Method that generates the initial values of the evolution model using"
        "the associated Markov Chain model                                    "
        
        # 2d array data from evolution model initial population attributes
        self.nij        = evoInit.nij           # array for genotype abundances
        self.mij        = np.zeros(self.nij.shape)

        # Note: mutation counts need to be initialized at 0 or greater for abs
        # fitness and 1 or greater for rel fitness
        self.bij_mutCnt = evoInit.bij_mutCnt    # b-terms mutation counts
        self.dij_mutCnt = evoInit.dij_mutCnt    # d-terms mutation counts
        self.cij_mutCnt = evoInit.cij_mutCnt    # c-terms mutation counts

        # set max sim time
        self.tmax       = evoInit.tmax          # max iterations
        
        # set the pfix solver method to 3 (use selection coeff) since pfix is not
        # needed to determine state space. Then generate the MC DRE model
        self.params['pfixSolver'] = 3
        self.mcModel = mcFac.mcFactory().createMcModel(self.params)

        # generate the fitness map for environmental changes
        self.fitMapEnv  = self.get_fitMapEnv()
        
        return None
    
    # --------------------------------------------------------------------------
    
    def get_populationMutations(self):
        # IMPLEMENTATION OF ABSTRACT METHOD
        "Method that defines mutations of genotypes. Mutations depend on the  "
        "the type of absolute fitness evoltuion model selected. This method   "
        "is best applied to the array form of abundances.                     "
        "                                                                     "
        "Absolute fitness mutations add/modify rows of the 2d evolution arrays"
        "while relative fitness mutations add columns to 2d evo arrays        "      
        
        # expand evo arrays to include new slots for mutations
        self.get_evoArraysExpand()

        # calculate the array with new juveniles, mij
        Ub = self.mcModel.params['Ua']
        Uc = self.mcModel.params['Uc']
        temp_mij    = self.get_mij() 
        final_mij   = np.zeros(temp_mij.shape)

        # Add beneficial absolute fitness trait mutation fluxes
        final_mij = (1-Ub-Uc) * temp_mij  # no mutations
        temp_mij[1:-1,:] = self.mcModel.params['Ua']*temp_

        # Add beneficial absolute fitness trait mutation fluxes
        
        return None

    
    #------------------------------------------------------------------------------
    
    def get_evoArraysExpand(self):
        # get_evoArraysExpand() expands evo arrays with additional rows/cols 
        # so that mutations can be added. Two key steps are:
        #   1. pads the set of array storing the evolutioanry state of the population
        #   2. fills in the padded the bij, dij and cij 
        #
        
        # expand arrays for abundances, birth, death and competition terms 
        self.nij = np.pad(self.nij,1)
        self.mij = np.pad(self.mij,1)

        # expand mutation count arrays, but these need more than padding to keep
        # mutation counts correct
        self.bij_mutCnt = self.get_mutationCountArrays(self.bij_mutCnt,'abs')
        self.dij_mutCnt = self.get_mutationCountArrays(self.dij_mutCnt,'')
        self.cij_mutCnt = self.get_mutationCountArrays(self.cij_mutCnt,'rel')

        # expand array for stochastic 
        self.stochThrsh = np.pad(self.stochThrsh,1)
        
        # lastly, we check if any of the bij_mutCnt are negative, in which case
        # we need to trim off the associated row
        bij_min = np.min(self.bij_mutCnt[0,:])
        
        if (bij_min < 0):
            self.nij = self.nij[1:-1,:]
            self.mij = self.mij[1:-1,:]
            
            self.bij_mutCnt = self.bij_mutCnt[1:-1,:]
            self.dij_mutCnt = self.dij_mutCnt[1:-1,:]
            self.cij_mutCnt = self.cij_mutCnt[1:-1,:]
            
            self.stochThrsh = self.stochThrsh[1:-1,:]
            
        return None

    # --------------------------------------------------------------------------
    
    def get_bij(self):
        # IMPLEMENTATION OF ABSTRACT METHOD
        "Method to calculate array of bij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        
        # get the list of b mutation counts
        bmc = self.bij_mutCnt[:,0]
        
        # map the mutation counts to bij values from the MC state space
        bij = [self.mcModel.bij[ii] for ii in bmc]
        bij = np.list(bij)        
        # 
        
        return bij
    
    # --------------------------------------------------------------------------
    
    def get_dij(self):
        # IMPLEMENTATION OF ABSTRACT METHOD
        "Method to calculate array of dij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "

        self.dij_mutCnt
        
        return None
    
    # --------------------------------------------------------------------------
    
    def get_cij(self):
        "Method to calculate array of cij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        
        return None
    
    
    #%% ------------------------------------------------------------------------
    # Specific class methods
    # --------------------------------------------------------------------------
    
    def get_fitMapEnv(self):
        # The method get_fitMapEnv checks each index and finds a prio index such 
        # that sum of sa[ii]+...+sa[ii-iBack] ~ fitness decline / iteration
        fitMap = []
        sa_i = self.mcModel.sa_i

        # environmental fitness decline per iteration
        se_per_iter = self.mcModel.params['se'] * self.mcModel.params['R']

        # loop through
        for ii in range(len(self.mcModel.sa_i)):
            # current state is ii, and we want to map back to iibk
            iBack = 0
            while ((ii-iBack >= 0) and (np.sum(sa_i[ii-iBack::ii+1]) < se_per_iter)):
                iBack = iBack+1
            fitMap.append(ii-iBack)

        return fitMap
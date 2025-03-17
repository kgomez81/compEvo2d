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
    # # Members of simClass 
    # self.params       # dictionary with evo parameters
    # self.absFitType   # absolute fitness evolution term
    # self.tmax         # max iterations (default is 10)
    
    # # 2d array data
    # self.nij              # array for genotype abundances
    # self.mij              # array for genotype abundances
    # self.bij_mutCnt       # array for genotype b-terms mutation counts
    # self.dij_mutCnt       # array for genotype d-terms mutation counts
    # self.cij_mutCnt       # array for genotype c-terms mutation counts
    
    # # - mcModel is associated Markov Chain used to reference state space
    # self.mcModel    
    # # - fitMapEnv maps fitness declines from environmental change across the state space
    # #   to try and achieve homogenous fitness declines across the state space. 
    # self.fitMapEnv     
    
    # # Simulation flag - 
    # #   true: stochastic birth/comp/death simulated, 
    # #   false: poisson sampling with expectation given dens-dep lottery model Eqtns
    # self.fullStochModelFlag 
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,simInit):
        
        # call constructor of base class
        super().__init__(simInit)
        
    #%% ------------------------------------------------------------------------
    # Definitions for abstract methods
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
        temp_mij    = self.get_mij_noMutants() 
        final_mij   = np.zeros(temp_mij.shape)

        # calculate the mutation fluxes
        # 1) no mutations
        final_mij = (1-Ub-Uc) * temp_mij  

        # 2) b mutations 
        final_mij[1:-1,:] = final_mij[1:-1,:] + self.mcModel.params['Ua'] * temp_mij[0:-2,:]

        # 3) c mutation
        final_mij[:,1:-1] = final_mij[:,1:-1] + self.mcModel.params['Uc'] * temp_mij[:,0:-2]
        
        self.mij = final_mij

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
        #   Note: we use same value for all classes for now unless the full 
        #         simulation with individual thresholds until a method is 
        #         developed based of selection coeff.
        self.stochThrsh = np.ones(self.nij.shape) * self.get_stochasticDynamicsCutoff()
        
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

    def run_environmentalDegredation(self):
        # Environmental degredation shifts classes back by ~se/t_e [df/iter]
        # Note: we assume that the padded nij has not been trimmed.

        if (self.simpleEnvShift):
            # simple environmental shift
            temp_nij = np.zeros(self.nij.shape)
            temp_nij[0:-2,:] = self.nij[1:-1,:]
        else:
            # complicated environmental shift
            
            # first we need an array large enought to account for the shifts 
            # back. get a list of class and map them to class after the shift 
            # back using the fitness map.
            bMutList = [self.bij_mutCnt[ii,0] for ii in range(self.bij_mutCnt.shape[0])]
            bMutTemp = [self.fitMapEnv[bNew] for bNew in bMutList]
            bMutShft = [ii for ii in range(min(bMutTemp),max(bMutTemp)+1)]

            # get new dimensions
            nb = len(bMutShft)
            nc = self.bij_mutCnt.shape[1]

            # now form the new evo arrays for mutation counts
            self.bij_mutCnt = np.tile(bMutShft,(nc,1)).T
            self.dij_mutCnt = np.tile(self.cij[0,:],(nb,1))
            self.cij_mutCnt = np.ones((nb,nc))

            # now shift abundances back using the fitness map
            temp_nij = np.zeros((nb,nc))
            temp_mij = np.zeros((nb,nc))

            for ii in range(len(bMutList)):
                i1 = bMutList[ii]
                i2 = bMutTemp[ii]
                temp_nij[i2,:] = temp_nij[i2,:] + self.nij[i1,:]
                temp_mij[i2,:] = temp_mij[i2,:] + self.mij[i1,:]

            # adjust stoch threshold array (simple version)
            self.stochThrsh = np.ones(self.nij.shape) * self.get_stochasticDynamicsCutoff()

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
        bij = np.list(bij,(self.bij_mutCnt.shape[1],1)).T
        
        return bij
    
    # --------------------------------------------------------------------------
    
    def get_dij(self):
        # IMPLEMENTATION OF ABSTRACT METHOD
        "Method to calculate array of dij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "

        # no mutations in dij values, so we use an array with a constant value
        dij = np.ones(self.dij_mutCnt.shape) * self.params['d']
        
        return dij
    
    # --------------------------------------------------------------------------
    
    def get_cij(self):
        "Method to calculate array of cij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        
        # get the list of c mutation counts
        cmc = self.cij_mutCnt[0,:]

        # map the mutation counts to cij values from the competition coefficient
        # definition: cj = (1+c+)**mutCnt
        cij = [(1+self.params['cp'])**ii for ii in cmc]
        cij = np.list(cij,(self.cij_mutCnt.shape[0],1))

        return cij

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
    
    #%% ------------------------------------------------------------------------
    # Specific class methods
    # --------------------------------------------------------------------------
    
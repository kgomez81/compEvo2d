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
        self.tmax       = 10                          # max iterations (default is 10)
        
        # 2d array with adult abundances
        self.nij        = np.zeros([1,1])       # array for genotype abundances (adults only)
        self.mij        = np.zeros([1,1])       # array for genotype mutant juventiles

        # Note: These are 2d arrays to store mutation counts, not actual bij, dij, cij values
        #       For actual values of bij, dij and cij, the mutation counts need to be mapped
        #       to an appropriate state space create from a Markov Chain class
        #
        #       Mutation count are as follows, 
        #       absolute fitness increases along the row dimension
        #       relative fitness increases along the column dimension
        #
        self.bij_mutCnt     = np.zeros([1,1])   # array for genotype b-terms mutation counts
        self.dij_mutCnt     = np.zeros([1,1])   # array for genotype d-terms mutation counts
        self.cij_mutCnt     = np.zeros([1,1])   # array for genotype c-terms mutation counts
        
        # - mcModel is associated Markov Chain used to reference state space
        self.mcModel    = []
        # - fitMapEnv maps fitness declines from environmental change across the state space
        #   to try and achieve homogenous fitness declines across the state space. 
        self.fitMapEnv  = np.zeros([1,1])   
        
        # Simulation flag - 
        #   true: stochastic birth/comp/death simulated, 
        #   false: poisson sampling with expectation given dens-dep lottery model Eqtns
        self.fullStochModelFlag = False         
        
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
    def get_evoArraysExpand(self):
        "Method that expand the evo arrays and updates mutation counts for    "
        "the bij, dij, cij arrays mutation counts.                            "
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
        # main function to run evolutionary model, iterations continue until
        # the population is extinct or tmax is reached.
        
        # set time index
        t           = 0
        popExtinct  = False 
        
        # run model tmax generations or until population is extinct
        while ( (t < self.tmax) and not popExtinct ):

            if not (self.fullStochModelFlag):

                # get new juveniles and modify arrays for potentially new classes
                # this will pad arrays and fill in an updated mij w/ mutation fluxes
                self.get_populationMutations()
                
                # run selection with Lotter model 
                self.run_determinsticEvolution()

                # run poisson samling of each class
                self.run_poissonSamplingOfAbundances()

            else:
                
                # get new juveniles and modify arrays for potentially new classes
                # this will pad arrays and fill in an updated mij w/ mutation fluxes
                self.get_populationMutations()

                # run competitino phase of model
                self.run_competitionPhase()
                
                # run death phase of model
                self.run_deathPhase()
            
            # collapse 2d array boundaries
            self.get_evoArraysCollapse()

            # update time increment
            t = t + 1
            
            popExtinct = (self.popSize() < 10)
        
        return None

    # Methods for Full Stochastic Model 
    #------------------------------------------------------------------------------
    # INCOMPLETE
    def run_competitionPhase(self):
        # run_competitionPhase() generates the set of juvenilees and 
        # simulates competition to determine the number of adults prior
        # to the death phase.
        
        # Note: get_populationMutations() should expand the arrays to capture 
        #       the appearance of juvenile mutants. How this occurs is unique to the 
        #       type of model (d vs b evo, RM vs DRE)
        # 
        
        # COMPETITION for STOCHASTIC CLASSES
        # 
        
        # COMPETITION for DETERMINISTIC CLASSES
        #
    
        return None
    
    #------------------------------------------------------------------------------
    
    def run_deathPhase(self):
        # run_deathPhase() gernates the set of new and remaining adults 
        # following death.
        
        # DEATH for DETERMINISTIC CLASSES
        # first calculate deterministic death of all classes
        temp_dij = self.get_dij()
        temp_nij = self.nij*(1/temp_dij)
        
        # DEATH for STOCHASTIC CLASSES
        # first get list of classes with stochastic dynamics
        stoch_ij = self.get_stochasticFitnessClasses()
        
        # apply death phase to stochastically modeled classes, by replacing 
        # the deterministic calculations from above
        for idx in stoch_ij:
            # sample deaths using binomial distribution
            temp_nij[idx[0],idx[1]] = np.random.binomial(self.nij[idx[0],idx[1]],1/temp_dij[idx[0],idx[1]],1)
        
        # replace nij with calculated values, but rounding to integer values
        self.nij = temp_nij - np.mod(temp_nij,1)
        
        return None

    #------------------------------------------------------------------------------
    
    def get_stochasticFitnessClasses(self):
        # get_stochasticFitnessClasses() identifies the set of genotypes that are 
        # below the stochastic threshold, whose abundances will not be modeled 
        # deterministically.
        
        # clear current list
        stoch_ij = []
        
        # calculate the stochastic threshold
        for ii in range(self.nij.shape[0]):
            for jj in range(self.nij.shape[1]):
                # only pick small nonzero classes
                if (self.nij[ii,jj] < self.stochThrsh) and (self.nij[ii,jj] > 0):
                    stoch_ij.append([ii,jj])
        
        return stoch_ij
    
    #------------------------------------------------------------------------------
    
    def get_arrayMap(self,A):
        # Used to flattens 2d arrays so that they can be used by SIM_functions
        # the map allows calculated values from functions to be placed back 
        # to a 2d array
        #
        # A will be either nij or mij, depending on the use of the map.
        ijMap = []
        kk    = 0

        # Iterate by rows first in order to have a correspondance with array flatten 
        for ii in range(A.shape[0]):
            for jj in range(A.shape[1]):
                if (A[ii,jj] > 0):
                    ijMap.append([ii,jj,kk])
                    kk = kk + 1
        
        return ijMap
    
    #------------------------------------------------------------------------------
    
    def get_mutationCountArrays(self,A,fitType):
        # The method get_mutationCountArrays pads mutation count arrays and 
        # increments the mutations counts for the new entries, depending on
        # whether the array is for abs or rel fitness mutations.

        mutCntSeq = [ii for ii in range(np.min(A)-1,np.max(A)+2)]
        
        if (fitType == 'abs'):
            # mutations increment along the row dimension
            temp_A = np.tile(mutCntSeq,1,A.shape[1]).T

        if (fitType == 'rel'):
            # mutations increment along the column dimension
            temp_A = np.tile(mutCntSeq,1,A.shape[0])

        else:
            temp_A = np.ones(np.pad(A,1).shape)

        return temp_A
    
    #------------------------------------------------------------------------------

    def get_evoArraysCollapse(self):
        # get_evoArraysCollapse() collapses evo arrays to remove any rows/cols
        # with only zero entries
        
        # get indices for rows and columns with only zero entries, excluding 
        # those within the bulk
        [idxR1,idxR2] = self.get_trimIndicesForZeroRows(self.nij)
        [idxC1,idxC2] = self.get_trimIndicesForZeroRows(self.nij.T)
        
        # trim arrays rows and columns with only zero entries, excluding those
        # within the bulk
        self.nij        = self.nij[idxR1:idxR2,idxC1:idxC2]
        self.mij        = self.mij[idxR1:idxR2,idxC1:idxC2]

        # Note: these array are trimmed here but their values must be maintined
        #       in the implementation of get_populationMutations()
        self.bij_mutCnt = self.bij_mutCnt[idxR1:idxR2,idxC1:idxC2]
        self.dij_mutCnt = self.dij_mutCnt[idxR1:idxR2,idxC1:idxC2]
        self.cij_mutCnt = self.cij_mutCnt[idxR1:idxR2,idxC1:idxC2]

        # Note: these array are trimmed here but their values must be maintined
        #       the implementation of get_populationMutations()
        self.stochThrsh = self.stochThrsh[idxR1:idxR2,idxC1:idxC2]
        
        return None
    
    #------------------------------------------------------------------------------
    
    def get_trimIndicesForZeroRows(self,A):
        # generic function to trim rows with zeros from top and bottom of A

        # sum across column dimension of A to get list of rows w/ nonzero entries
        nzR_l2f = np.sign(np.sum(A,1))
        nzR_f2l = nzR_l2f[::]

        # find first nonzero row beginning from 0 index
        idx1 = np.argmax(nzR_l2f)
        
        # find first nonzero row beginning from last index downward
        idx2 = len(nzR_f2l) - np.argmax(nzR_f2l)
        
        return [idx1,idx2]
    
    #------------------------------------------------------------------------------
    
    def get_stochasticDynamicsCutoff(self):
        # calc_StochThreshold determines abundance for establishment of class
        # to be used in full and approximate simulation.
        
        # temporary value
        nijEst = 100       
        
        # establishment based on abs/rel pFix values
        
        return nijEst
    
    # Methods for Approximate Stochastic Model 
    # --------------------------------------------------------------------------

    def run_determinsticEvolution(self):

        # First flatten nij, mij, etc. to calculate the expected abundances using 
        # lotter model functions. To do this, we use get_arrayMap to see which 
        # array ii,jj entries are nonzero with resepect to mij array. The third 
        # entry of idxMap entry is corresponding index of ii,jj to flattened array.
        idxMap  = self.get_arrayMap(self.mij)
        idxkk   = [ii[2] for ii in idxMap]

        mij_f = self.mij.flatten()[idxkk]
        cij_f = self.mij.flatten()[idxkk]

        # calculate the expected number of new adults
        delta_nij_plus = lmFun.deltnplus(mij_f,cij_f,U)

        # now add the expected new adults to the nij array (nij + delta_nij_plus)
        for idx in range(len(idxkk)):
            crnt_ii = idxMap[idx][0]
            crnt_jj = idxMap[idx][1]
            self.nij[crnt_ii,crnt_jj] = self.nij[crnt_ii,crnt_jj] + delta_nij_plus[idx]

        # apply death phase to complete calculation of lottery model selection
        #
        #    n_ij(t+1) = (1/dij) * (nij + delta_nij+)
        # 
        self.nij = (1/self.dij) * self.nij

        return None
    
    #------------------------------------------------------------------------------

    def run_poissonSamplingOfAbundances(self):
        # we run a simple poisson sampling for abundances with mean equal to 
        # values calculated from the density-dependent lottery model computations
        self.nij = np.random.poisson(self.nij,self.nij.shape)

        return None

    # General Methods
    #------------------------------------------------------------------------------
    
    def get_bbar(self):
        # get_bbar() returns population mean value of the b-terms
        #    Note: requires implementation of bij
        
        bbar = np.sum(self.nij*self.get_bij())/self.popSize()
        
        return bbar
    
    #------------------------------------------------------------------------------
    
    def get_dbar(self):
        # get_dbar() returns population mean value of the d-terms
        #    Note: requires implementation of dij
        
        dbar = np.sum(self.nij*self.get_dij())/self.popSize()
        
        return dbar
    
    #------------------------------------------------------------------------------
    
    def get_cbar(self):
        # get_cbar() returns population mean value of the c-terms
        #    Note: requires implementation of cij

        cbar = np.sum(self.nij*self.get_cij())/self.popSize()
        
        return cbar
    
    #------------------------------------------------------------------------------
    
    def get_U(self):    
        # get_U() returns the number of unoccupied territories
        
        U = self.params['T']-self.popSize()    
        
        return U
    
    #------------------------------------------------------------------------------
    
    def get_mij_noMutants(self):
        # get_mij() returns the expected juveniles, but does not include mutations
        
        mij = self.nij*self.get_bij()*(self.get_U()/self.params['T'])
        
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
    
    def clear_juvenilePop_mij(self):

        # reset all juvenile counts to zero
        self.mij = self.mij*0

        return None

    #------------------------------------------------------------------------------
    # INCOMPLETE
    def output_evoStats(self):

        # calculate averages

        return None
    #------------------------------------------------------------------------------
    # INCOMPLETE
    def store_evoSnapshot():
        
        # function to store evolutionary state, and continue simulations
        # - time
        # - nij, bij, cij, dij        
        # - snapshot of parameters (as csv file)
        
    
        return None
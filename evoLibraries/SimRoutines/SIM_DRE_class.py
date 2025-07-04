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

import numpy as np

import evoLibraries.SimRoutines.SIM_class as sim
import evoLibraries.evoExceptions as evoExc

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
    # self.modelDynamics 
    
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
        
        # if mutation rates are zero, then its not a travelling wave
        NoMutations = (self.params['Ua'] == 0) and (self.params['Uc'] == 0)
        NoEnvChange = (self.params['R'] == 0) or (self.params['se'] == 0)
        if (NoMutations and NoEnvChange): 
            self.mij  = self.get_mij_noMutants() 
            return None
        
        # expand evo arrays to include new slots for mutations
        self.get_evoArraysExpand()

        # calculate the array with new juveniles, mij
        temp_mij    = self.get_mij_noMutants() 

        # calculate the mutation fluxes
        # 1) no mutations
        final_mij   = (1-self.params['Ua']-self.params['Uc']) * temp_mij

        # 2) b mutations 
        final_mij[1:,:] = final_mij[1:,:] + self.params['Ua'] * temp_mij[0:-1,:]

        # 3) c mutation
        final_mij[:,1:] = final_mij[:,1:] + self.params['Uc'] * temp_mij[:,0:-1]
        
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
            # Simple environmental shift model
            #
            # We assum that environmental degredation shifts the population back
            # one state leading to  ~ sa_i per shift, but we vary the probability
            # of the occurrence to achieve a desired ve. We use the mean abs
            # fitness to decide how to vary the rate.
            #
            self.bij_mutCnt -= 1
            
        else:
            # complicated environmental shift
            # 
            # The probability of a shift back is constant, so the shifts back
            # have to be approximately ~ se, so abundances have to be shuffuled 
            # across the state space to achieve this
            
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
        bmc = [int(self.bij_mutCnt[int(ii),0]) for ii in range(self.bij_mutCnt.shape[0])]
        
        # map the mutation counts to bij values from the MC state space
        bij = [self.mcModel.bi[ii] for ii in bmc]
        bij = np.tile(bij,(self.bij_mutCnt.shape[1],1)).T
        
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
        # IMPLEMENTATION OF ABSTACT METHOD
        "Method to calculate array of cij actual values from mutation counts. "
        "The calculation is model specific, i.e. b vs d evo, RM vs DRE.       "
        
        # get the list of c mutation counts
        cmc = [int(self.cij_mutCnt[0,int(ii)]) for ii in range(self.cij_mutCnt.shape[1])]
        
        # map the mutation counts to cij values from the competition coefficient
        # definition: cj = (1+c+)**mutCnt
        cij = [(1+self.params['cp'])**ii for ii in cmc]
        cij = np.tile(cij,(self.cij_mutCnt.shape[0],1))

        return cij

    # --------------------------------------------------------------------------

    def get_fitMapEnv(self):
        # IMPLEMENTATION OF ABSTACT METHOD
        " The method get_fitMapEnv checks each index and finds a prio index such "
        " that sum of sa[ii]+...+sa[ii-iBack] ~ fitness decline / iteration      "
        fitMap = []
        sa_i = self.mcModel.sa_i

        # environmental fitness decline per event
        se_per_iter = self.params['se']

        # loop through
        for ii in range(len(self.mcModel.sa_i)):
            # current state is ii, and we want to map back to iibk
            iBack = 0
            while ((ii-iBack >= 0) and (np.sum(sa_i[ii-iBack::ii+1]) < se_per_iter)):
                iBack = iBack+1
            fitMap.append(int(ii-iBack))

        return fitMap
    
    # --------------------------------------------------------------------------
    
    def get_saij(self):
        # IMPLEMENTATION OF ABSTACT METHOD
        " The method returns the absolute fitness selection coefficents       "
        
        # get the list of b mutation counts
        bmc = [int(self.bij_mutCnt[int(ii),0]) for ii in range(self.bij_mutCnt.shape[0])]
        
        # map the mutation counts to bij values from the MC state space
        sbij = [self.mcModel.sa_i[ii] for ii in bmc]
        sbij = np.tile(sbij,(self.bij_mutCnt.shape[1],1)).T

        return sbij
    
    # --------------------------------------------------------------------------
    
    def get_sabar(self):
        # IMPLEMENTATION OF ABSTACT METHOD
        " The method returns the mean absolute fitness selection coefficents  "
        
        sbbar = np.sum(self.nij*self.get_saij())/np.sum(self.nij)

        return sbbar
    
    # --------------------------------------------------------------------------
    
    def get_ibarAbs(self):
        # IMPLEMENTATION OF ABSTACT METHOD
        " The method returns the mean state over the absolute fitness space   "
        
        ibarAbs = np.sum(self.nij*self.bij_mutCnt)/np.sum(self.nij)

        return ibarAbs
    
    # --------------------------------------------------------------------------
    
    def get_saEnvShift(self):
        # IMPLEMENTATION OF ABSTACT METHOD
        " The method returns the sa fitness increment one state back, which   "
        " is needed to caclulate the rate of environmental change.            "
        
        # get the state behind the rounded back mean state
        
        ibarBack = int(np.floor(self.get_ibarAbs()-1))
        
        saEnvShift = self.mcModel.sa_i[ibarBack]     

        return saEnvShift
    
    # --------------------------------------------------------------------------
    
    def get_projected_nijDistrition(self,fitType):
        # IMPLEMENTATION OF ABSTACT METHOD
        " get_projected_nijDistrition returns the distribiton of abundances   "
        " along the desired fitness dimension                                 "
        
        try: 
            if (fitType == 'abs'):
                # sum abundances along the rel fit dimension (sum along cols = 1)
                proj_nij = np.sum(self.nij,1)  
                # get idx for abs fitness dim 
                idxs_mut = self.bij_mutCnt[:,0]
            elif (fitType == 'rel'):
                # sum abundances along the abs fit dimention (sum along rows = 0)
                proj_nij = np.sum(self.nij,0)
                # get idx for rel fitness dim 
                idxs_mut = self.cij_mutCnt[0,:]
            else:
                raise evoExc.EvoInvalidInput('Invalide Fitness Type Provided', custom_kwarg=fitType)
                
        except evoExc.EvoInvalidInput as exc:
            print(f'Fitness type must be string: abs or rel, but method was provided {exc.custom_kwarg}')
        
        return [proj_nij,idxs_mut]
    
    # #%% ------------------------------------------------------------------------
    # # SIM class inherited methods
    # # --------------------------------------------------------------------------
    
    # #---------------------------------------------------------------------------
    # # Class methods
    # # --------------------------------------------------------------------------
    
    # def run_evolutionModel(self):
    #     # main function to run evolutionary model, iterations continue until
    #     # the population is extinct or tmax is reached.

    # # --------------------------------------------------------------------------
    # # Methods for Full Stochastic Model 
    # #------------------------------------------------------------------------------
    
    # def run_competitionPhase(self):
    #     # run_competitionPhase() generates the set of juvenilees and 
    #     # simulates competition to determine the number of adults prior
    #     # to the death phase.
    
    # #------------------------------------------------------------------------------
    
    # def run_deathPhase(self):
    #     # run_deathPhase() gernates the set of new and remaining adults 
    #     # following death.

    # #------------------------------------------------------------------------------
    
    # def get_stochasticFitnessClasses(self):
    #     # get_stochasticFitnessClasses() identifies the set of genotypes that are 
    #     # below the stochastic threshold, whose abundances will not be modeled 
    #     # deterministically.
    
    # #------------------------------------------------------------------------------
    
    # def get_arrayMap(self,A):
    #     # Used to flattens 2d arrays so that they can be used by SIM_functions
    #     # the map allows calculated values from functions to be placed back 
    #     # to a 2d array
    
    # #------------------------------------------------------------------------------
    
    # def get_mutationCountArrays(self,A,fitType):
    #     # The method get_mutationCountArrays pads mutation count arrays and 
    #     # increments the mutations counts for the new entries, depending on
    #     # whether the array is for abs or rel fitness mutations.
    
    # #------------------------------------------------------------------------------

    # def get_evoArraysCollapse(self):
    #     # get_evoArraysCollapse() collapses evo arrays to remove any rows/cols
    #     # with only zero entries
    
    # #------------------------------------------------------------------------------
    
    # def get_trimIndicesForZeroRows(self,A):
    #     # generic function to trim rows with zeros from top and bottom of A
    
    # #------------------------------------------------------------------------------
    
    # def get_stochasticDynamicsCutoff(self):
    #     # calc_StochThreshold determines abundance for establishment of class
    #     # to be used in full and approximate simulation.
    
    # # --------------------------------------------------------------------------
    # # Methods for Approximate Stochastic Model 
    # # --------------------------------------------------------------------------

    # def run_determinsticEvolution(self):
    #     # main method to run the model
    
    # #------------------------------------------------------------------------------

    # def run_poissonSamplingOfAbundances(self):
    #     # we run a simple poisson sampling for abundances with mean equal to 
    #     # values calculated from the density-dependent lottery model computations

    # # --------------------------------------------------------------------------
    # # General Methods
    # #------------------------------------------------------------------------------
    
    # def get_bbar(self):
    #     # get_bbar() returns population mean value of the b-terms
    #     #    Note: requires implementation of bij
    
    # #------------------------------------------------------------------------------
    
    # def get_dbar(self):
    #     # get_dbar() returns population mean value of the d-terms
    #     #    Note: requires implementation of dij
    
    # #------------------------------------------------------------------------------
    
    # def get_cbar(self):
    #     # get_cbar() returns population mean value of the c-terms
    #     #    Note: requires implementation of cij
    
    # #------------------------------------------------------------------------------
    
    # def get_U(self):    
    #     # get_U() returns the number of unoccupied territories
    
    # #------------------------------------------------------------------------------
    
    # def get_mij_noMutants(self):
    #     # get_mij() returns the expected juveniles, but does not include mutations
    
    # #------------------------------------------------------------------------------
    
    # def get_lij(self):
    #     # get_lij() returns the lottery model lij parameters for each genotype, 
    #     # which are the poisson juvenile birth rates over unoccupied territories 
    
    # #------------------------------------------------------------------------------
    
    # def get_L(self):
    #     # get_L() returns the lottery model L parameter (sum of the poisson birth 
    #     # parameters over non-occupied territories)
    
    # #------------------------------------------------------------------------------
    
    # def popsize(self):
    #     # get_popsize() returns the total population size 
        
    # #------------------------------------------------------------------------------
    
    # def clear_juvenilePop_mij(self):
    #     # reset all juvenile counts to zero

    # #------------------------------------------------------------------------------
    
    # def get_qi(self,idx,fitType):
    #     # return the theoretical travelling wave width

    # #------------------------------------------------------------------------------
    
    # def get_covAbsRel(self):
    #     # The method returns the absolute & relative fitness covariance 

    # # --------------------------------------------------------------------------
    
    # def get_varAbs(self):
    #     # The method returns the absolute fitness variance 
    
    # # --------------------------------------------------------------------------
    
    # def get_varRel(self):
    #     # The method returns the relative fitness variance 
        
    # # --------------------------------------------------------------------------
    
    # def get_scij(self):
    #     # The method returns the relative fitness selection coefficients
    
    # # --------------------------------------------------------------------------
    
    # def get_scbar(self):
    #     # The method returns the mean relative fitness selection coefficients

    # # --------------------------------------------------------------------------
    
    # def get_idx_nijMode(self,fitType):
    #     # get_idx_nijMode returns the b/c mutation index for mode of abundances
    #     # along fitType dimension (abs or rel). If there are two classes that 
    #     # have identical abundances, then first index is regarded as the mode.

    # # --------------------------------------------------------------------------
    
    # def get_ibarRel(self):
    #     # The method returns the mean state over the relative fitness space.
    #     # this is ibar with renormalized mutation counts, not absolute mutation
    #     # counts.

    # #------------------------------------------------------------------------------
    
    # def get_veIter(self):
    #     # calculates the rate of environmental degredation per iteration

    # #------------------------------------------------------------------------------

    # def get_lambdaEnvPoiss(self):
    #     # calculates poisson rate of environental change per iter to achieve
    #     # an approximate desired ve.
    #     # 
    #     # 1. SIMPLE MODEL (shift back by one)
    #     # this is done by varying the probability of an event, instead of the 
    #     # fitness decrease. we use the mean fitness as the reference; i.e. find
    #     # and equivalent number of small steps to one large one, or vice versa.
    #     #
    #     # Caculate the poiss rate by finding "lam_env"
    #     # 
    #     #      ve = sa(ibbar-1) / E[T | lam_env] = sa(ibbar-1)* lam_env
    #     # 
    #     # This leads to setting a poss rate of 
    #     #
    #     #      lam_env = ve / sa(ibbar-1) = (se / sa(ibbar-1))/ Te
    #     #
    #     # where R = 1/Te. This will achieve ve at the bulk.
    #     #
    #     # 2. COMPLEX MODE (shift abundances across state space)
    #     # We keep the sampling rate of environmental degredation constant, and
    #     # reshuffule abundances in the state space to achieve ve.

    # #------------------------------------------------------------------------------

    # def sample_environmentalDegredation(self):
    #     # sample_environmentalDegredation checks if an environmental 
    #     # degredation event has occured and if so, it calls the method 
    #     # run_environmentalDegredation

    # #------------------------------------------------------------------------------

    # def output_evoStats(self,ti):
    #     # The method output_evoStats will collect all of the mean values of bij, cij,
    #     # dij, and popsize
    
    # #------------------------------------------------------------------------------
    
    # def store_evoSnapshot(self):       
    #     #  Capture the final snapshot, use the simInit class features and save
    #     # a new simInit drived from the prior one

    # #------------------------------------------------------------------------------

    # def check_adaptiveEvent(self):
    #     # check_adaptiveStatistics is ran each iteration to determine if there
    #     # has been an adaptive event, i.e. the mean fitness state incremented
    #     # 
    #     # cases where mean fitness has dropped because of environmental changes
    #     # are handled in sample_environmentalDegredation()
    
    # #------------------------------------------------------------------------------
    
    # def store_adpativeEventStatistics(self):
    #     # store_adpativeEventStatistics() calculates the outputs to track the 
    #     # statistics of adaptive events. 
        
    # #------------------------------------------------------------------------------
    
    # def output_selectionDyanmics(self,ti):
    #     # The method output_selectionDyanmics will output abundances as a time
    #     # series. This can only be used when mutation rates are zero, and env
    #     # change is absent.

    # #------------------------------------------------------------------------------
    
    # def check_evoStop(self,t):
    #     # The method check_evoStop checks key condtions to detemrine if evolution 
    #     # should stop
    
    # #------------------------------------------------------------------------------
    
    # def maxMcState(self):
    #     # check if the max MC state has been reached
    
    # #------------------------------------------------------------------------------
    
    # def get_adaptiveEventsLogFilename(self):
    #     # get_adaptiveEventsLogFilename() is a simple wrapper to get the name
    #     # of the file with the logged adaptive events. 
    
    # #------------------------------------------------------------------------------
    
    # def get_selectionDynamicsFilename(self):
    #     # get_selectionDynamicsFilename() is a simple wrapper to get the name
    #     # of the file that stores the selection dynamics.
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

import numpy as np
import os

from abc import ABC, abstractmethod
# from joblib import Parallel, delayed, cpu_count

import evoLibraries.LotteryModel.LM_functions as lmFun
import pickle
import time

class simClass(ABC):
    # ABSTRACT class used to carry out evolution simulations
    #
    # IMPORTANT: This class is not mean to be used for anything other then for 
    # consolidating the common code of RM and DRE SIM classes, i.e. something like 
    # an Abstract class
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,simInit):
        # simInit is class with initial values for evo arrays below.
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        # - mcModel is associated Markov Chain used to reference state space
        self.mcModel = simInit.mcModel
        self.params  = self.mcModel.params       # take the copy for the MC model

        self.tmax       = simInit.tmax           # max iterations
        self.tcap       = simInit.tcap           # number of iterations between each snapshot
        
        # 2d array with adult abundances
        self.nij        = simInit.nij               # array for genotype abundances (adults only)
        self.mij        = np.zeros(self.nij.shape)  # array for genotype mutant juventiles 

        # Note: These are 2d arrays to store mutation counts, not actual bij, dij, cij values
        #       For actual values of bij, dij and cij, the mutation counts need to be mapped
        #       to an appropriate state space create from a Markov Chain class
        #
        #       Mutation count are as follows, 
        #       absolute fitness increases along the row dimension
        #       relative fitness increases along the column dimension
        #
        self.bij_mutCnt = simInit.bij_mutCnt    # b-terms mutation counts
        self.dij_mutCnt = simInit.dij_mutCnt    # d-terms mutation counts
        self.cij_mutCnt = simInit.cij_mutCnt    # c-terms mutation counts
        
        # - fitMapEnv maps fitness declines from environmental change across the state space
        #   to try and achieve homogenous fitness declines across the state space. 
        self.fitMapEnv  = self.get_fitMapEnv() 
        
        # Simulation flag - 
        #   true: stochastic birth/comp/death simulated, 
        #   false: poisson sampling with expectation given dens-dep lottery model Eqtns
        self.modelDynamics      = simInit.modelDynamics
        self.simpleEnvShift     = simInit.simpleEnvShift
        
        # stochastic dynamics cutoff
        # use single cutoff for entire set for now.
        self.stochThrsh = np.ones(self.nij.shape) * self.get_stochasticDynamicsCutoff()

        # file output paramters
        self.simDatDir              = '/'.join((simInit.outputsPath,simInit.simDatDir))
        # filenames for two types of outputs
        # eg. outputStatsFile = ''
        self.outputStatsFileBase    = simInit.outputStatsFileBase
        self.outputSnapshotFileBase = simInit.outputSnapshotFileBase

        # we need alternates for file outputs of particular runs, this will ensure
        # runs don't overwrite prior data
        tempT = time.localtime()
        datetimeStamp = "%d%02d%02d_%02d%02d%02d" % (tempT.tm_year,tempT.tm_mon,tempT.tm_mday,tempT.tm_hour,tempT.tm_min,tempT.tm_sec)

        self.outputStatsFile    = self.outputStatsFileBase.replace('.csv','_'+datetimeStamp+'.csv')
        self.outputSnapshotFile = self.outputSnapshotFileBase.replace('.pickle','_'+datetimeStamp+'.pickle')
        
        # make the output dir
        if not (os.path.exists(self.simDatDir)):
            # if the directory does not exist then generate it
            os.mkdir(self.simDatDir)

        # lastly, we keep a copy of simInit to use for the final snapshot
        self.simInit = simInit
        
        # counter for environmental shifts
        self.envShiftCntr = 0
        
        # mutation axis minimum to track total adaptation in relative fitness 
        # integers track the minimum mutation count
        self.ibarRel_MinMutCnt = 0
        
        # Adaptive events logger, with entries corresponding to following
        #
        #  0. time of last adaptive event
        #  1. current absolute fitness state
        #  2. rate correction factor (kappa)
        #  3. adaptive event counter
        #
        # One key assumption is va_mean = va_front as measure and so we meaure
        # va_mean as an estimate of the rate of adaptation. The rate correction 
        # factor helps adjust sojourn times following environmental changes. 
        # For example, if ibar_Abs = i + kappa < i + 1 at the time of an event
        # degrading the environment by 1, then we cannot simply use the number
        # of iterations it takes to get from i-1+kappa -> i as the sojourn time.
        # So, we use instead
        #
        #    Est. sojourn time =  # iterations (i-1+kappa => i) / (1-kappa)
        #
        # with the assumption being that va_mean = d_ibar/dt ~ const.
        
        # Initialize for time zero, and current mean fitness, no kappa correction
        self.adaptiveEventAbs_Log = dict({'last_event_time':0, 
                                     'crnt_fitness_state': np.floor(self.get_ibarAbs()),
                                     'sojourn_kappa': 0,
                                     'adapt_event_counter':0 })
        
        # Adatpive events relative fitness logger, logs adaptive events in 
        # relative fitness for a given absolute fitness state. 
        #
        # Similar to the absolute fitness case, we simply log the a start time 
        # of the prior event or environmental degredation event, log a kappa 
        # correction factor to account for cases with the latter. 
        #
        # However, in this case we have to keep track of the associated absolute
        # fiteness state space because that indicates what the strength of 
        # selection is for the relative trait by virtue of its density dependence
        self.adaptiveEventRel_Log = dict({'last_event_time':0, 
                                     'crnt_fitness_state': np.floor(self.get_ibarRelTot()),
                                     'sojourn_kappa': 0,
                                     'adapt_event_counter':0,
                                     'current_abs_state': np.floor(self.get_ibarAbs())})
        # sim time
        self.t = 0
        
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
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_fitMapEnv(self):
        "Method generates map that helps apply appropriate absolute fitness   "
        "close to the provided value and rate in paramters                    "
        pass

    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_saij(self):
        " The method returns the absolute fitness selection coefficents       "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_sabar(self):
        " The method returns the mean absolute fitness selection coefficents  "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_ibarAbs(self):
        " The method returns the mean state over the absolute fitness space   "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_saEnvShift(self):
        " The method returns the sa fitness increment one state back, which   "
        " is needed to caclulate the rate of environmental change.            "
        pass
    
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_projected_nijDistrition(self,fitType):
        " get_projected_nijDistrition returns the distribiton of abundances   "
        " along the desired fitness dimension                                 "
        pass
    
    #%% ------------------------------------------------------------------------
    # Class methods
    # --------------------------------------------------------------------------
    
    def run_evolutionModel(self):
        # main function to run evolutionary model, iterations continue until
        # the population is extinct or tmax is reached.
        
        # run model tmax generations or until population is extinct
        while ( self.continue_evoSim() ):
            
            if (self.modelDynamics == 0):

                # get new juveniles and modify arrays for potentially new classes
                # this will pad arrays and fill in an updated mij w/ mutation fluxes
                self.get_populationMutations()

                # run competitino phase of model
                self.run_competitionPhase()
                
                # run death phase of model
                self.run_deathPhase()
                
            elif (self.modelDynamics == 1):
                
                # get new juveniles and modify arrays for potentially new classes
                # this will pad arrays and fill in an updated mij w/ mutation fluxes
                self.get_populationMutations()
                
                # run selection with Lotter model 
                self.run_determinsticEvolution()
                
            else:
                # get new juveniles and modify arrays for potentially new classes
                # this will pad arrays and fill in an updated mij w/ mutation fluxes
                self.get_populationMutations()
                
                # run selection with Lotter model 
                self.run_determinsticEvolution()
                
                # run poisson samling of each class
                self.run_poissonSamplingOfAbundances()
                
            # check if we should if environment has changed
            self.sample_environmentalDegredation()
            
            # Collapse the array for Abundances.
            # NOTE: collapse has to occur before we output evo stats
            self.get_evoArraysCollapse()
            
            # check to see if a snapshot of the mean needs to be taken
            if (self.tcap>0) and (np.mod(self.t,self.tcap)==0):
                self.output_evoStats()
                
                # for runs with selection dynamics, take snapshot of abundances
                if (self.modelDynamics == 1):
                    self.output_selectionDyanmics()
            
            # update time increment
            self.t = self.t + 1
            
            # after iteration, check if adaptive event occurred and log it
            self.check_adaptiveEvent()
        
        # store final results for end of simulation
        self.store_evoSnapshot()

        return None

    # --------------------------------------------------------------------------
    # Methods for Full Stochastic Model 
    #------------------------------------------------------------------------------
    
    def run_competitionPhase(self):
        # run_competitionPhase() generates the set of juvenilees and 
        # simulates competition to determine the number of adults prior
        # to the death phase.
        # 
        # !!!!!!!! NOT IMPLEMENTED !!!!!!!!!!!!!!!!!!!
        #
        idxMap  = self.get_arrayMap(self.mij)
        idxkk   = [ii[2] for ii in idxMap]

        mij_f = self.mij.flatten()[idxkk]
        cij_f = self.mij.flatten()[idxkk]

        # calculate the expected number of new adults
        delta_nij_plus = lmFun.deltnplus(mij_f,cij_f,self.get_U())

        # now add the expected new adults to the nij array (nij + delta_nij_plus)
        for idx in range(len(idxkk)):
            crnt_ii = idxMap[idx][0]
            crnt_jj = idxMap[idx][1]
            self.nij[crnt_ii,crnt_jj] = self.nij[crnt_ii,crnt_jj] + delta_nij_plus[idx]
    
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
                if (self.nij[ii,jj] < self.stochThrsh[ii,jj]) and (self.nij[ii,jj] > 0):
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
        i1 = int(np.min(A))-1
        i2 = int(np.max(A))+2
        mutCntSeq = [ii for ii in range(i1,i2)]
        
        if (fitType == 'abs'):
            # mutations increment along the row dimension
            temp_A = np.tile(mutCntSeq,(A.shape[1]+2,1)).T

        elif (fitType == 'rel'):
            # mutations increment along the column dimension
            temp_A = np.tile(mutCntSeq,(A.shape[0]+2,1))

        else:
            temp_A = self.dij_mutCnt[0,0]*np.ones((self.dij_mutCnt.shape[0]+2,self.dij_mutCnt.shape[1]+2))

        return temp_A
    
    #------------------------------------------------------------------------------

    def get_evoArraysCollapse(self):
        # get_evoArraysCollapse() collapses evo arrays to remove any rows/cols
        # with only zero entries
        
        # if mutation rates are zero, then its not a travelling wave
        NoMutations = (self.params['Ua'] == 0) and (self.params['Uc'] == 0)
        NoEnvChange = (self.params['R'] == 0) or (self.params['se'] == 0)
        if (NoMutations and NoEnvChange): 
            return None
        
        # get indices for rows and columns with only zero entries, excluding 
        # those within the bulk
        idxR = self.get_trimIndicesForZeroRows(self.nij)
        idxC = self.get_trimIndicesForZeroRows(self.nij.T)
        
        # trim arrays rows and columns with only zero entries, excluding those
        # within the bulk
        self.nij        = self.nij[idxR[0]:idxR[1],idxC[0]:idxC[1]]
        self.mij        = self.mij[idxR[0]:idxR[1],idxC[0]:idxC[1]]

        # Note: these array are trimmed here but their values must be maintined
        #       in the implementation of get_populationMutations()
        self.bij_mutCnt = self.bij_mutCnt[idxR[0]:idxR[1],idxC[0]:idxC[1]]
        self.dij_mutCnt = self.dij_mutCnt[idxR[0]:idxR[1],idxC[0]:idxC[1]]
        self.cij_mutCnt = self.cij_mutCnt[idxR[0]:idxR[1],idxC[0]:idxC[1]]
        
        # need to renormalize the cij mutation counts to prevent cij values 
        # from growing too large; but save the scale adjustment to track total
        # adaptation on the relative fitness axis.
        #
        # e.g.  We adjust as follows
        #
        #       cij_mutCnt  =   [2 3] -->   [0 1]
        #                       [2 3]       [0 1]
        #
        #       ibarRelTot            --> ibarRelTot + 2
        #
        self.ibarRel_MinMutCnt = self.ibarRel_MinMutCnt + np.min(self.cij_mutCnt)
        self.cij_mutCnt = self.cij_mutCnt-np.min(self.cij_mutCnt)

        # Note: these array are trimmed here but their values must be maintined
        #       the implementation of get_populationMutations()
        self.stochThrsh = self.stochThrsh[idxR[0]:idxR[1],idxC[0]:idxC[1]]
        
        return None
    
    #------------------------------------------------------------------------------
    
    def get_trimIndicesForZeroRows(self,A):
        # generic function to trim rows with zeros from top and bottom of A

        # sum across column dimension of A to get list of rows w/ nonzero entries
        As = np.sign(np.sum(A,1))
        
        # array for non-zero entries
        nzR = np.array([ii for ii in range(len(As))])  
        nzR = nzR[As>0]
        
        # find first nonzero row beginning from 0 index
        idx1 = np.min(nzR)
        
        # find first nonzero row beginning from last index downward
        idx2 = np.max(nzR)+1
        
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
        
        cij_f = self.get_cij()
        cij_f = cij_f.flatten()[idxkk]

        # calculate the expected number of new adults
        delta_nij_plus = lmFun.deltnplus(mij_f,cij_f,self.get_U())
        
        # now add the expected new adults to the nij array (nij + delta_nij_plus)
        for idx in range(len(idxkk)):
            crnt_ii = idxMap[idx][0]
            crnt_jj = idxMap[idx][1]
            self.nij[crnt_ii,crnt_jj] = self.nij[crnt_ii,crnt_jj] + delta_nij_plus[idx]
        
        # apply death phase to complete calculation of lottery model selection
        #
        #    n_ij(t+1) = (1/dij) * (nij + delta_nij+)
        # 
        self.nij = (1/self.get_dij()) * self.nij
        
        return None
    
    #------------------------------------------------------------------------------

    def run_poissonSamplingOfAbundances(self):
        # we run a simple poisson sampling for abundances with mean equal to 
        # values calculated from the density-dependent lottery model computations
        
        temp_nij = self.nij
        
        for ii in range(temp_nij.shape[0]):
            for jj in range(temp_nij.shape[1]):
                if (temp_nij[ii,jj]>0) and (temp_nij[ii,jj] < 1e5):
                    temp_nij[ii,jj] = np.random.poisson(temp_nij[ii,jj],1)
                else:
                    temp_nij[ii,jj] = round(temp_nij[ii,jj])
        
        self.nij = temp_nij

        return None

    # --------------------------------------------------------------------------
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
    
    def popSize(self):
        # get_popsize() returns the total population size 
        
        popsize = np.sum(self.nij)
        
        return popsize
    
    #------------------------------------------------------------------------------
    
    def clear_juvenilePop_mij(self):

        # reset all juvenile counts to zero
        self.mij = self.mij*0

        return None
    
    #------------------------------------------------------------------------------
    
    def get_qi(self,idx,fitType):
        # return the theoretical travelling wave width
        
        if (fitType == 'abs'):
            s = self.mcModel.sa_i[idx]
            U = self.mcModel.Ua_i[idx]
        else:
            s = self.mcModel.sc_i[idx]
            U = self.mcModel.Uc_i[idx]
        N = self.mcModel.eq_Ni[idx]
        
        qi = 2*np.log(N*s)/np.log(s/U)
        
        return qi
    
    # --------------------------------------------------------------------------
    
    def get_covAbsRel(self):
        # The method returns the absolute & relative fitness covariance 
        # 
        # Note: due to size of values, the calculations are scaled by sbbar
        #       and scbar, so that they can be printed reasonably in outputs 
        #       i.e. we calculate
        #
        #       cov_scaled = 1/(sbbar*scbar)*cov(sb,sc)
        
        delta_sa = (self.get_saij()/self.get_sabar() - 1) # Equals (sa - sabar)/sabar
        delta_sc = (self.get_scij()/self.get_scbar() - 1) # Equals (sc - scbar)/scbar
        
        covAbsRel = np.sum(self.nij * delta_sa * delta_sc)/np.sum(self.nij)
        
        return covAbsRel
    
    # --------------------------------------------------------------------------
    
    def get_varAbs(self):
        # The method returns the absolute fitness variance 
        # 
        # Note: due to size of values, the calculations are scaled by sbbar
        #       and scbar, so that they can be printed reasonably in outputs 
        #       i.e. we calculate
        #
        #       var_scaled = 1/(sbbar**2)*var(sb)
        
        delta_sa = (self.get_saij()/self.get_sabar() - 1) # Equals (sb - sbbar)/sbbar
        
        varAbs = np.sum(self.nij * delta_sa * delta_sa)/np.sum(self.nij)

        return varAbs
    
    # --------------------------------------------------------------------------
    
    def get_varRel(self):
        # The method returns the relative fitness variance 
        # 
        # Note: due to size of values, the calculations are scaled by sbbar
        #       and scbar, so that they can be printed reasonably in outputs 
        #       i.e. we calculate
        #
        #       var_scaled = 1/(sbbar**2)*var(sb)
        
        delta_sc = (self.get_scij()/self.get_scbar() - 1) # Equals (sc - scbar)/scbar
        
        varRel = np.sum(self.nij * delta_sc * delta_sc)/np.sum(self.nij)

        return varRel

    # --------------------------------------------------------------------------
    
    def get_scij(self):
        # The method returns the relative fitness selection coefficients
        
        cmc = [int(self.cij_mutCnt[0,int(ii)]) for ii in range(self.cij_mutCnt.shape[1])]
        
        # map the mutation counts to cij values from the competition coefficient
        # definition: cj = (1+c+)**mutCnt
        scij = [self.mcModel.sc_i[ii] for ii in cmc]
        scij = np.tile(scij,(self.cij_mutCnt.shape[0],1))

        return scij
    
    # --------------------------------------------------------------------------
    
    def get_scbar(self):
        # The method returns the mean relative fitness selection coefficients

        scbar = np.sum(self.nij*self.get_scij())/np.sum(self.nij)

        return scbar
        
    # --------------------------------------------------------------------------
    
    def get_idx_nijMode(self,fitType):
        # get_idx_nijMode returns the b/c mutation index for mode of abundances
        # along fitType dimension (abs or rel). If there are two classes that 
        # have identical abundances, then first index is regarded as the mode.
        
        nij_proj_dstr = self.get_projected_nijDistrition(fitType)
        
        proj_nij = nij_proj_dstr[0]
        
        idx_mode = np.argmax(proj_nij)
        
        return idx_mode
    
    # --------------------------------------------------------------------------
    
    def get_ibarRel(self):
        # The method returns the mean state over the relative fitness space.
        # this is ibar with renormalized mutation counts, not absolute mutation
        # counts.
        
        ibarRel = np.sum(self.nij*self.cij_mutCnt)/np.sum(self.nij)

        return ibarRel
    
    # --------------------------------------------------------------------------
    
    def get_ibarRelTot(self):
        # The method returns the mean state over the relative fitness space.
        # this is ibar is for absolute mutation counts
        
        ibarRelTot = np.sum(self.nij*self.cij_mutCnt)/np.sum(self.nij) + self.ibarRel_MinMutCnt

        return ibarRelTot
    
    #------------------------------------------------------------------------------
    
    def get_veIter(self):
        # calculates the rate of environmental degredation per iteration
        
        # get ve per generation 
        veIter = self.params['se']*self.params['R']
        
        return veIter
    
    #------------------------------------------------------------------------------
    
    def get_lambdaEnvPoiss(self):
        # calculates poisson rate of environental change per iter to achieve
        # an approximate desired ve.
        # 
        # 1. SIMPLE MODEL (shift back by one)
        # this is done by varying the probability of an event, instead of the 
        # fitness decrease. we use the mean fitness as the reference; i.e. find
        # and equivalent number of small steps to one large one, or vice versa.
        #
        # Caculate the poiss rate by finding "lam_env"
        # 
        #      ve = sa(ibbar-1) / E[T | lam_env] = sa(ibbar-1)* lam_env
        # 
        # This leads to setting a poss rate of 
        #
        #      lam_env = ve / sa(ibbar-1) = (se / sa(ibbar-1))/ Te
        #
        # where R = 1/Te. This will achieve ve at the bulk.
        #
        # 2. COMPLEX MODE (shift abundances across state space)
        # We keep the sampling rate of environmental degredation constant, and
        # reshuffule abundances in the state space to achieve ve.
        
        if (self.simpleEnvShift):
            # get the fitness index and increment one step back from the bulk
            # saEnvShift = sa(ibbar-1) above.
            saEnvShift = self.get_saEnvShift()
        else:
            saEnvShift = self.params['se']    
        
        # calculate the poisson rate with fitness adjustment
        lam_env = self.get_veIter()/saEnvShift
        
        return lam_env
    
    #------------------------------------------------------------------------------
    
    def sample_environmentalDegredation(self):
        # sample_environmentalDegredation checks if an environmental 
        # degredation event has occured and if so, it calls the method 
        # run_environmentalDegredation
        
        # special sim code for cases where environmental decay is disabled
        # since the state space of our MC model is limited to the max state that 
        # is calculated, we have to reset the model to the initial state when

        # first check if we are evolving without changes in the environment
        # and apply degredation to reset to the initial state
        if ((self.params['R']==0) or (self.params['se']==0)) and self.maxMcState():
            # reset the states to moved back towards the initial one saved in simInit
            envDegrCnt = np.max(self.bij_mutCnt) - np.max(self.simInit.bij_mutCnt) + 1
        else:
            # if we're evolving with 
            envDegrCnt = np.random.poisson(self.get_lambdaEnvPoiss())
            self.envShiftCntr += envDegrCnt
        
        if (envDegrCnt>0):
            # we could get multiple shifts back at once, in which case
            # we depart a bit from ve, but not by much.
            while(envDegrCnt > 0):
                # shift the distribution back accordingly
                self.run_environmentalDegredation()  
                envDegrCnt = envDegrCnt-1
                
            # reset log for adaptive events, due to environment changing
            self.reset_adaptiveEventAbs_Log('EnvEvent')
            
            # also need to reset the log for adaptive events in relative fitness
            # when environmental events occur because we can only properly measure
            # them at a fixed state in absolute fitness space. When absolute
            # fitnes changes, the log must be adjusted 
            self.reset_adaptiveEventRel_Log('EnvEvent')
        
        return None
    
    #------------------------------------------------------------------------------
    
    def output_evoStats(self):
        # The method output_evoStats will collect all of the mean values of bij, cij,
        # dij, and popsize, as well as other outputs 
        
        # setup o parameters for data collection
        outputs = []
        headers = []
        
        idx_bmod = self.get_idx_nijMode('abs') # abs fitness state for mode
        biCrrnt  = self.get_bij()

        idx_bmax = np.max(self.bij_mutCnt[:,0])
        idx_bbar = self.get_ibarAbs()
        
        idx_bmod_int = self.bij_mutCnt[idx_bmod,0]
        idx_bmax_int = int(idx_bmax)
        idx_bbar_int = int(np.round(idx_bbar))
        
        # -------------------------------------
        # population 
        outputs.append(self.t)                                      # 00. time  
        outputs.append(np.sum(self.nij))                            # 01. popsize
        outputs.append(np.sum(self.nij)/self.params['T'])           # 02. gamma

        headers.append('time')
        headers.append('popsize')
        headers.append('gamma')

        # -------------------------------------
        # b-mutation data
        outputs.append(np.min(self.get_bij()))                      # 03. min bi
        outputs.append(np.max(self.get_bij()))                      # 04. max bi
        outputs.append(self.get_bbar())                             # 05. mean b
        outputs.append(biCrrnt[idx_bmod,0])                         # 06. mode b
        outputs.append(np.min(self.bij_mutCnt))                     # 07. b-index min
        outputs.append(np.max(self.bij_mutCnt))                     # 08. b-index max
        outputs.append(idx_bbar_int)                                # 09. b-index mean
        outputs.append(idx_bmod_int)                                # 10. b-index mode
        outputs.append(self.get_qi(idx_bbar_int,'abs'))             # 11. b-index qi
        
        headers.append('min_bi')
        headers.append('max_bi')
        headers.append('mean_bi')
        headers.append('mode_bi')
        headers.append('min_b_idx')
        headers.append('max_b_idx')
        headers.append('mean_b_idx')
        headers.append('mode_b_idx')
        headers.append('qi_bavg')
        
        # -------------------------------------
        # c-mutation data
        outputs.append(np.min(self.get_cij()))                      # 12. min cj
        outputs.append(np.max(self.get_cij()))                      # 13. max cj
        outputs.append(self.get_cbar())                             # 14. mean c
        outputs.append(np.min(self.cij_mutCnt))                     # 15. c-index min 
        outputs.append(np.max(self.cij_mutCnt))                     # 16. c-index max
        outputs.append(self.get_ibarRel())                          # 17. c-index mean 
        outputs.append(self.get_qi(idx_bbar_int,'rel'))             # 18. c-index qi 
        
        headers.append('min_ci')
        headers.append('max_ci')
        headers.append('mean_ci')
        headers.append('min_c_idx')
        headers.append('max_c_idx')
        headers.append('mean_c_idx')
        headers.append('qi_cavg')
        
        # -------------------------------------
        # variance/covariance fitness values
        # note: these are scaled by the mean selection coefficients (last 2 entries)
        outputs.append(self.get_varAbs())                           # 19. var_abs
        outputs.append(self.get_varRel())                           # 20. var_rel
        outputs.append(self.get_covAbsRel())                        # 21. covAbsRel
        outputs.append(self.get_sabar())                            # 22. mean sa_i
        outputs.append(self.get_scbar())                            # 23. mean sc_i
        
        headers.append('var_abs')
        headers.append('var_rel')
        headers.append('covAbsRel')
        headers.append('mean_sa')
        headers.append('mean_sc')

        # -------------------------------------
        # rate of adaptation 
        # note: for vc, we capture the vc at the average and max abs fitness state
        outputs.append(self.mcModel.va_i[idx_bmax_int])             # 24. va at imaxAbs
        outputs.append(self.mcModel.va_i[idx_bbar_int])             # 25. va at rounded ibarAbs
        outputs.append(self.mcModel.vc_i[idx_bmax_int])             # 26. vc at imaxAbs
        outputs.append(self.mcModel.vc_i[idx_bbar_int])             # 27. vc at rounded ibarAbs
        
        headers.append('va_imax')
        headers.append('va_ibar')
        headers.append('vc_imax')
        headers.append('vc_ibar')

        # -------------------------------------
        # selection coefficients
        outputs.append(self.mcModel.sa_i[idx_bmax_int])             # 28. sa at imaxAbs
        outputs.append(self.mcModel.sa_i[idx_bbar_int])             # 29. sa at rounded ibarAbs
        outputs.append(self.mcModel.sc_i[idx_bmax_int])             # 30. sc at imaxAbs
        outputs.append(self.mcModel.sc_i[idx_bbar_int])             # 31. sc at rounded ibarAbs
        
        headers.append('sa_imax')
        headers.append('sa_ibar')
        headers.append('sc_imax')
        headers.append('sc_ibar')

        # -------------------------------------
        # mutation rates
        outputs.append(self.mcModel.Ua_i[idx_bmax_int])             # 32. Ua at imaxAbs
        outputs.append(self.mcModel.Ua_i[idx_bbar_int])             # 33. Ua at rounded ibarAbs
        outputs.append(self.mcModel.Uc_i[idx_bmax_int])             # 34. Uc at imaxAbs
        outputs.append(self.mcModel.Uc_i[idx_bbar_int])             # 35. Uc at rounded ibarAbs

        headers.append('Ua_imax')
        headers.append('Ua_ibar')
        headers.append('Uc_imax')
        headers.append('Uc_ibar')
        
        # -------------------------------------
        outputs.append(np.min(self.get_dij()))                      # 36. d-term
        outputs.append(self.envShiftCntr)                           # 37. environment shifts
        headers.append('d_term')
        headers.append('envShft')
        
        # open the file and append new data
        if (self.t==0):  
            # if logging for first time, add headers             
            self.log_evoData(self.outputStatsFile,outputs,headers)
        else:
            # subsequent logs don't require headers
            self.log_evoData(self.outputStatsFile,outputs)
        
        return None
    
    #------------------------------------------------------------------------------
    
    def store_evoSnapshot(self):
        
        # To capture the final snapshot, we use the simInit class features and save
        # a new simInit drived from the prior one
        simInitSave = self.simInit

        # replace the init arrays with last copies of nij, bij_mutCnt, etc.
        simInitSave.nij = self.nij
        simInitSave.bij_mutCnt = self.bij_mutCnt
        simInitSave.dij_mutCnt = self.dij_mutCnt
        simInitSave.cij_mutCnt = self.cij_mutCnt

        # save the data to a pickle file
        with open(self.outputSnapshotFile, 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(simInitSave, file)

        return None

    #------------------------------------------------------------------------------
    
    def check_adaptiveEvent(self):
        # check_adaptiveStatistics is ran each iteration to determine if there
        # has been an adaptive event, i.e. the mean fitness state incremented
        # 
        # Note:
        # cases where mean fitness has dropped because of environmental changes
        # are handled in sample_environmentalDegredation() for absolute fitness
        #
        # Overall process:
        #
        # 1. Check for types of adaptive events
        # 2. Output the adaptation events
        # 3. Reset the adaptation logs
        #
        
        # was there an adaptation in the absolute fitness trait
        adapEventAbs_Flag = self.adaptiveEventAbs_Log['crnt_fitness_state'] < np.floor(self.get_ibarAbs())
        
        # was there an adaptation in the relative fitness trait
        adapEventRel_Flag = self.adaptiveEventRel_Log['crnt_fitness_state'] < np.floor(self.get_ibarRelTot())
        
        # Check the different cases where mean fitness jumped to the next state 
        if ( adapEventAbs_Flag and not (adapEventRel_Flag) ):
            
            # log the adaptive event
            self.output_adpativeEventStatistics('abs')
            
            # reset the log
            self.reset_adaptiveEventAbs_Log('AbsEvent')
        
        elif ( not (adapEventAbs_Flag) and adapEventRel_Flag ):
            
            # log the adaptive event
            self.output_adpativeEventStatistics('rel')
            
            # reset the logs
            self.reset_adaptiveEventRel_Log('RelEvent')
            
        elif ( adapEventAbs_Flag and adapEventRel_Flag ):
            
            # log the adaptive event
            self.output_adpativeEventStatistics('abs')
            self.output_adpativeEventStatistics('absrel')
            
            # reset the logs
            self.reset_adaptiveEventAbs_Log('AbsEvent')
            self.reset_adaptiveEventAbs_Log('AbsRelEvent')
            
        return None
    
    #------------------------------------------------------------------------------
    
    def reset_adaptiveEventAbs_Log(self,eventType):
        # simple function to reset the adaptive events log for absolute
        # fitness
        
        # regarless of the event, we always need to update
        #
        #   - last_event_time
        #   - crnt_fitness_state
        #
        # the other fields are case dependent
        #
        #   - sojourn_kappa
        #   - adapt_event_counter
        #
        self.adaptiveEventAbs_Log['last_event_time']        = self.t
        self.adaptiveEventAbs_Log['crnt_fitness_state']     = np.floor(self.get_ibarAbs())
        
        if (eventType == 'EnvEvent'):
            # for enviromental degredation events, we don't update the counter
            # for absolute adaptations, just set a new kappa for sojourn time
            # adjustments in post processing.
            self.adaptiveEventAbs_Log['sojourn_kappa'] = np.mod(self.get_ibarAbs(),1)           # set kappa after fitness decline
            
        elif (eventType == 'AbsEvent'):
            # for absolute fitness adaptation events, increment the counter 
            # and reset the sojourn kappa
            self.adaptiveEventAbs_Log['sojourn_kappa']          = 0
            self.adaptiveEventAbs_Log['adapt_event_counter']    = self.adaptiveEventAbs_Log['adapt_event_counter']+1
            
        return None
    
    #------------------------------------------------------------------------------
    
    def reset_adaptiveEventRel_Log(self,eventType):
        # simple function to reset the adaptive events log for relative
        # fitness
        
        # regarless of the event, we always need to update
        #
        #   - last_event_time
        #   - crnt_fitness_state
        #
        # the other fields are case dependent
        #
        #   - sojourn_kappa
        #   - adapt_event_counter
        #   - current_abs_state
        #
        self.adaptiveEventRel_Log['last_event_time']        = self.t
        self.adaptiveEventRel_Log['crnt_fitness_state']     = np.floor(self.get_ibarRelTot())
        
        if (eventType == 'EnvEvent') or (eventType == 'AbsEvent'):
            
            # for environmental degredation, we have to reset the aboslute 
            # fitness state field, and sojourn kappa            
            self.adaptiveEventRel_Log['sojourn_kappa']          = np.mod(self.get_ibarRelTot(),1) # set kappa after fitness decline
            self.adaptiveEventRel_Log['current_abs_state']      = np.floor(self.get_ibarAbs())
            
        elif (eventType == 'RelEvent'):
            
            # for the case when there is an adaptation in relative fitness, we 
            # increment the counter, and reset the sojourn kappa value
            self.adaptiveEventRel_Log['sojourn_kappa']          = 0
            self.adaptiveEventRel_Log['adapt_event_counter']    = self.adaptiveEventRel_Log['adapt_event_counter']+1
        
        elif (eventType == 'AbsRelEvent'):
            
            # Special where an absolute and relative adaptive event occurs; in
            # these cases we also update the current_abs_state as minus 1 since
            # abs_state was previously incremented prior to this function call.
            self.adaptiveEventRel_Log['sojourn_kappa']          = 0
            self.adaptiveEventRel_Log['adapt_event_counter']    = self.adaptiveEventRel_Log['adapt_event_counter']+1
            self.adaptiveEventRel_Log['current_abs_state']      = np.floor(self.get_ibarAbs())
            
        return None
    
    #------------------------------------------------------------------------------
    
    def output_adpativeEventStatistics(self,fitType):
        # store_adpativeEventStatistics() calculates the outputs to track the 
        # statistics of adaptive events. 
        #
        # We used identical headers for absolute and relative fitness files
        #
        #  1. fitness_state: relative or absolute fitness state of the 
        #                    measured sojourn time.
        #
        #  2. sojourn_time:  iterations spent in the fitness_state (in 1) for
        #                    absolute and relative fitness
        #
        #  3. sojourn_kappa: scale adjustment for sojourn time to account for 
        #                    only partial times spent in a given state
        #
        #  4. adapt_counter: sadf
        #
        #  5. crnt_abs_fit:  This value only matters for relative fitness, and 
        #                    is used to indicate the associated abs fit state
        #                    for the measured sojourn time.
        
        # setup o parameters for data collection
        outputs = []
        headers = []
        
        if (fitType == 'abs'):
            
            sjrn_kappa = self.adaptiveEventAbs_Log['sojourn_kappa']
            fitn_state = self.adaptiveEventAbs_Log['crnt_fitness_state']
            adap_cntr  = self.adaptiveEventAbs_Log['adapt_event_counter']    
            crnt_iabs  = 0   # this value is irrelavent for va estimates
            
            # environment degredation can cause shorter then expected sojourn
            # times, which need to be corrected. We use the simplest correction
            # below, but also store kappa to filter out cases where sojourn 
            # times are unreliable.
            if (1 - self.adaptiveEventAbs_Log['sojourn_kappa'] > 0):
                # use Est. sojourn time =  # iterations (i-1+kappa => i) / (1-kappa)
                sjrn_time = (self.t-self.adaptiveEventAbs_Log['last_event_time'])/(1-sjrn_kappa)
            else:
                # alpha was set to 1 (should be no greater than 1)
                sjrn_time = (self.t-self.adaptiveEventAbs_Log['last_event_time'])
            
            # select the log files for absolute fitness
            logFile = self.get_adaptiveEventsLogFilename('abs')
            
        elif (fitType == 'rel') or (fitType == 'absrel'):
            sjrn_kappa = self.adaptiveEventRel_Log['sojourn_kappa']
            fitn_state = self.adaptiveEventRel_Log['crnt_fitness_state']
            adap_cntr  = self.adaptiveEventRel_Log['adapt_event_counter']    
            
            if (fitType == 'rel'):
                crnt_iabs  = self.adaptiveEventRel_Log['current_abs_state']
            else:
                crnt_iabs  = self.adaptiveEventRel_Log['current_abs_state']-1
            
            # environment degredation can cause shorter then expected sojourn
            # times, which need to be corrected. We use the simplest correction
            # below, but also store kappa to filter out cases where sojourn 
            # times are unreliable.
            if (1 - self.adaptiveEventRel_Log['sojourn_kappa'] > 0):
                # use Est. sojourn time =  # iterations (i-1+kappa => i) / (1-kappa)
                sjrn_time = (self.t-self.adaptiveEventRel_Log['last_event_time'])/(1-sjrn_kappa)
            else:
                # alpha was set to 1 (should be no greater than 1)
                sjrn_time = (self.t-self.adaptiveEventRel_Log['last_event_time'])
            
            # select the log files for absolute fitness
            logFile = self.get_adaptiveEventsLogFilename('rel')
            
        outputs.append(fitn_state)
        outputs.append(sjrn_time)
        outputs.append(sjrn_kappa)
        outputs.append(adap_cntr)
        outputs.append(crnt_iabs)
        headers.append('fitness_state')
        headers.append('sojourn_time')
        headers.append('sojourn_kappa')
        headers.append('adapt_counter')
        headers.append('crnt_abs_fit')
            
        # open the file and append new data
        if (adap_cntr==0):  
            # if logging for first time, add headers             
            self.log_evoData(logFile,outputs,headers)
        else:
            # subsequent logs don't require headers
            self.log_evoData(logFile,outputs)

        return None
    
    #------------------------------------------------------------------------------
    
    def output_selectionDyanmics(self):
        # The method output_selectionDyanmics will output abundances as a time
        # series. This can only be used when mutation rates are zero, and env
        # change is absent.
        
        # first check conditions to output selection dynamics
        NoMutations = (self.params['Ua'] == 0) and (self.params['Uc'] == 0)
        NoEnvChange = (self.params['R'] == 0 or self.params['se'] == 0)
        if not (NoMutations and NoEnvChange): return None
        
        # setup o parameters for data collection
        outputs = []
        headers = []
        
        outputs.append(self.t)
        headers.append('t')
        
        nij = self.nij.flatten()
        pij = (np.ones(nij.shape)*nij)/sum(nij)
        
        # output abundances
        for ii in range(nij.shape[0]):
            outputs.append(nij[ii])
            headers.append(("n%02d" % (ii)))
            
        # output frequencies
        for ii in range(pij.shape[0]):
            outputs.append(pij[ii])
            headers.append(("p%02d" % (ii)))
        
        # output selection coefficients
        idx_bbar     = self.get_ibarAbs()
        idx_bbar_int = int(np.round(idx_bbar))
        
        outputs.append(self.mcModel.sa_i[idx_bbar_int])
        outputs.append(self.mcModel.sc_i[idx_bbar_int])
        outputs.append(self.get_bbar())
        outputs.append(idx_bbar)
        outputs.append(np.mean(self.get_dij()))
        
        headers.append('sa')
        headers.append('sc')
        headers.append('mean_bi')
        headers.append('mean_b_idx')
        headers.append('d_term')
        
        # open the file and append new data
        if (self.t==0):   
            # if logging for first time, add headers             
            self.log_evoData(self.get_selectionDynamicsFilename(),outputs,headers)
        else:
            # subsequent logs don't require headers
            self.log_evoData(self.get_selectionDynamicsFilename(),outputs)
        
        return None

    #------------------------------------------------------------------------------
    
    def continue_evoSim(self):
        # The method check_evoStop checks key condtions to detemrine if evolution 
        # should stop
        max_t_not_hit   = ( self.t < self.tmax)
        pop_not_extinct = (self.popSize() > 10)
        
        stopEvolution = max_t_not_hit and pop_not_extinct
        
        # for sim runs with env change, don't stop for invalid mc state, as we 
        # we will reset the state space
        if (self.params['R'] == 0) or (self.params['se'] == 0):
            validMcState    = True
        else:
            validMcState    = not self.maxMcState()
        
        stopEvolution = max_t_not_hit and pop_not_extinct and validMcState
        
        return stopEvolution

    #------------------------------------------------------------------------------
    
    def maxMcState(self):
        # check if the max MC state has been reached
        maxMcState = np.argmax(self.mcModel.state_i)-2
        crntMcState = np.max(self.bij_mutCnt)
        return (crntMcState >= maxMcState)

    #------------------------------------------------------------------------------
    
    def get_adaptiveEventsLogFilename(self,fitType=None):
        # get_adaptiveEventsLogFilename() is a simple wrapper to get the name
        # of the file with the logged adaptive events.
        
        if (fitType == None) or (fitType == 'abs'):
            logFile = self.outputStatsFile.replace('.csv','_adaptLogAbs.csv')
        else:
            logFile = self.outputStatsFile.replace('.csv','_adaptLogRel.csv')
        
        return logFile
    
    #------------------------------------------------------------------------------
    
    def get_selectionDynamicsFilename(self):
        # get_selectionDynamicsFilename() is a simple wrapper to get the name
        # of the file that stores the selection dynamics. 
        return self.outputStatsFile.replace('.csv','_selDyn.csv')
    
    #------------------------------------------------------------------------------
    
    def get_evoSimFilename(self):
        # get_evoSimFilename() is a simple wrapper to get a standardized pickle 
        # output filename to save snapshots of the evoSim object in the same
        # directory as other file names.
        
        return self.outputSnapshotFile.replace('.pickle','_evoSim.pickle')
    
    #------------------------------------------------------------------------------
    
    def log_evoData(self,outputFile,outputs,headers=None):
        # log_evoData() is is a generic function to output data to files while 
        # the simulation runs
        
        # open the file and append new data
        with open(outputFile, "a") as file:
            if (headers!=None):
                # output column if at initial time
                file.write( ','.join(tuple(headers))+'\n' )
            
            # output data collected
            file.write( (','.join(tuple(['%.10f']*len(outputs))) + '\n') % tuple(outputs))
        
        return None
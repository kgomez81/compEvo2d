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
        self.params  = self.mcModel.params

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
        self.fullStochModelFlag = simInit.fullStochModelFlag
        self.simpleEnvShift     = simInit.simpleEnvShift
        
        # stochastic dynamics cutoff
        # use single cutoff for entire set for now.
        self.stochThrsh = np.ones(self.nij.shape) * self.get_stochasticDynamicsCutoff()

        # file output paramters
        self.simDatDir            = simInit.simDatDir
        # filenames for two types of outputs
        # eg. outputStatsFile = ''
        self.outputStatsFileBase    = simInit.outputStatsFileBase
        self.outputSnapshotFileBase = simInit.outputSnapshotFileBase

        # we need alternates for file outputs of particular runs, this will ensure
        # runs don't overwrite prior data
        tempT = time.localtime()
        datetimeStamp = "%d%02d%02d_%02d%02d" % (tempT.tm_year,tempT.tm_mon,tempT.tm_mday,tempT.tm_hour,tempT.tm_min)

        self.outputStatsFile    = self.outputStatsFileBase.replace('.csv','_'+datetimeStamp+'.csv')
        self.outputSnapshotFile = self.outputSnapshotFileBase.replace('.pickle','_'+datetimeStamp+'.pickle')
        
        # make the output dir
        if not (os.path.exists(self.simDatDir)):
            # if the directory does not exist then generate it
            os.mkdir(self.simDatDir)

        # lastly, we keep a copy of simInit to use for the final snapshot
        self.simInit = simInit
        
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
            # print('---------------')
            # print("time=%d" % (t))
            # print('---------------')
            
            if not (self.fullStochModelFlag):

                # get new juveniles and modify arrays for potentially new classes
                # this will pad arrays and fill in an updated mij w/ mutation fluxes
                self.get_populationMutations()
                
                # run selection with Lotter model 
                self.run_determinsticEvolution()
                
                # print('pre-poiss pop array')
                # print(self.nij)
                # run poisson samling of each class
                self.run_poissonSamplingOfAbundances()
                
                # print('post-poiss pop array')
                # print(self.nij)
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
            
            # print('post-collapse pop array')
            # print(self.nij)
            
            # print('cij mut array')
            # print(self.get_cij())
            # print('average c')
            # print(self.get_cbar())
            
            # check if we should if environment has changed
            sampleEnvDegrCnt = np.random.poisson(self.mcModel.params['R'])
            if (sampleEnvDegrCnt>0):
                while(sampleEnvDegrCnt > 0):
                    self.run_environmentalDegredation()
                    sampleEnvDegrCnt=sampleEnvDegrCnt-1
            
            # check to see if a snapshot of the mean needs to be taken
            if (self.tcap>0) and (np.mod(t,self.tcap)==0):
                self.output_evoStats(t)

            # update time increment
            t = t + 1
            
            # print('sizes')
            # print(self.nij.shape)
            # print(self.bij_mutCnt.shape)
            # print(self.cij_mutCnt.shape)
            
            popExtinct = (self.popSize() < 10)
        
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
        # temporarily not implemented 
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
        # from growing too large
        self.cij_mutCnt = self.cij_mutCnt-np.min(self.cij_mutCnt)+1

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
        nzR= np.array([ii for ii in range(len(As))])  
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
        
        # print('-- pre add n+ --')
        # print('2. pop array')
        # print(self.nij)
        # print('3. juveniles')
        # print(self.mij)
        # print('4. mij array')
        # print(mij_f)
        # # print('5. cij mutations')
        # # print(cij_f)
        # print('5. bij array')
        # print(self.get_bij())
        # print('6. calculated new adults')
        # print(idxMap)
        # print(delta_nij_plus)
        
        # now add the expected new adults to the nij array (nij + delta_nij_plus)
        for idx in range(len(idxkk)):
            crnt_ii = idxMap[idx][0]
            crnt_jj = idxMap[idx][1]
            self.nij[crnt_ii,crnt_jj] = self.nij[crnt_ii,crnt_jj] + delta_nij_plus[idx]
            
        # print('-- post add delta_n+ --')
        # print(self.nij)
        
        # apply death phase to complete calculation of lottery model selection
        #
        #    n_ij(t+1) = (1/dij) * (nij + delta_nij+)
        # 
        self.nij = (1/self.get_dij()) * self.nij
        
        # print('-- post add n+ and 1/d --')
        # print('1. pop array')
        # print(self.nij)
        # print('2. 1/dij array')
        # print(1/self.get_dij())
        
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
        
        U = self.mcModel.params['T']-self.popSize()    
        
        return U
    
    #------------------------------------------------------------------------------
    
    def get_mij_noMutants(self):
        # get_mij() returns the expected juveniles, but does not include mutations
        
        mij = self.nij*self.get_bij()*(self.get_U()/self.mcModel.params['T'])
        
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
    
    def output_evoStats(self,ti):
        # The method output_evoStats will collect all of the mean values of bij, cij,
        # dij, and popsize

        # collect outputs
        outputs = []
        outputs.append(ti)                                          # 00. time  
        outputs.append(np.sum(self.nij))                            # 01. popsize
        outputs.append(np.sum(self.nij)/self.mcModel.params['T'])   # 02. gamma
        outputs.append(self.get_ibar())                             # 03. ibar
        
        outputs.append(np.min(self.bij_mutCnt[:,0]))                # 04. min bi
        outputs.append(self.get_bbar())                             # 05. mean b
        outputs.append(np.max(self.bij_mutCnt[:,0]))                # 06. max bi
        outputs.append(np.sum(np.sign(np.sum(self.nij,1))))         # 07. b_width
        outputs.append(self.get_varAbs())                           # 08. var_abs
        
        outputs.append(np.min(self.cij_mutCnt[0,:]))                # 09. min cj
        outputs.append(self.get_cbar())                             # 10. mean c
        outputs.append(np.max(self.cij_mutCnt[0,:]))                # 11. max cj
        outputs.append(np.sum(np.sign(np.sum(self.nij,0))))         # 12. c_width
        outputs.append(self.get_varRel())                           # 13. var_rel
        
        outputs.append(self.get_covAbsRel())                        # 14. covAbsRel

        # open the file and append new data
        with open(self.outputStatsFile, "a") as file:
            if (ti==0):
                # output column if at initial time
                file.write("time,popsize,gamma,ibar_abs,min_b,avg_b,max_b,width_b,var_abs,min_c,avg_c,max_c,width_c,var_rel,covAbsRel\n")
            # output data collected
            file.write("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % tuple(outputs))
        
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
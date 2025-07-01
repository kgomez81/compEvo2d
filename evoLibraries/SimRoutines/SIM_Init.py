# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:09:32 2024

@author: Owner
"""

import os

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_factory as mcFac

#%% Main class for rho plots

class SimEvoInit():
    # class to automate the process of generating data for rho figures.
    
    #% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,simPathsIO,simArryData):
        # --------------------------------------------------------------------------
        # fields to specify input and output paths
        # --------------------------------------------------------------------------
        # Format for simPathsIO dictionary
        #   - outputStat
        #   - outputSnapshot
        #   - inputFile : parameter file for MC model

        # standard paths for inputs and outputs in the figure scripts directory.
        self.inputsPath  = os.path.join(os.getcwd(),'inputs')
        self.outputsPath = os.path.join(os.getcwd(),'outputs')
        
        # parameter file for MC model
        self.paramFile   = simPathsIO['paramFile']  # parameter file for MC model
        self.paramTag    = simPathsIO['paramTag']   # param file tag to idetify input for outputs
        self.paramFilePath = os.path.join(self.inputsPath,self.paramFile) # full path to input file

        # set output paths for data
        self.simDatDir   = simPathsIO['simDatDir']  # directory for outputs
        self.simDatFile1 = simPathsIO['statsFile']  # file name for stats print outs
        self.simDatFile2 = simPathsIO['snpshtFile'] # file name for end-of-run snapshot

        # joins below build the paths + file names for saving stats and snapshot of sim runs
        self.outputStatsFileBase    = \
            ''.join(('/'.join((self.outputsPath,self.simDatDir,'_'.join((self.simDatFile1,self.paramTag)))),'.csv'))
        self.outputSnapshotFileBase = \
            ''.join(('/'.join((self.outputsPath,self.simDatDir,'_'.join((self.simDatFile2,self.paramTag)))),'.pickle'))
        
        # --------------------------------------------------------------------------
        # Setup of parameters and MC model
        # --------------------------------------------------------------------------
        # Some key definitions: 
        #  - modelDyanmics - flag to indicate the expected type of model dynamics for
        #                    for mutation. Either:
        #                    0: full stochastics dynamics (no implemented)
        #                    1: detrministic model with/without mutations, and 
        #                    environmental changes.
        #                    !{0,1}: lottery model of selection with Poisson sampling
        #
        #  - simpleEnvShft - Either: 
        #                    1) for simple shifts back with rates scaled to achieve 
        #                    desired rate of environmental degredation (variable R), or 
        #                    2) for shifts back that shuffle abudances to decrease
        #                    fitness of individiuals by fixed amount (fixed s)
        #
        #  - modeType   - RM (running our of mutations) or DRE (diminishing returns
        #                 epistasis) for selection coefficients
        #
        # - absFitType - bEvo or dEvo for evolution in b- or d- trait
        #                 (dEvo sim class not implemented)
        #
        self.modelDynamics = simPathsIO['modelDynamics']
        self.simpleEnvShift     = simPathsIO['simpleEnvShift']
        self.modelType  = simPathsIO['modelType']
        self.absFitType = simPathsIO['absFitType']

        # generate the MC model 
        #   note: parameters are capture as a member of this class
        #   note: set pfixSolver type to 3 (use selection coeff) for faster calculations
        #         since we don't actually need pfix for the simulations
        self.evoOptions = evoObj.evoOptions(self.paramFilePath,self.modelType,self.absFitType)
        self.evoOptions.params['pfixSolver'] = 3
        
        self.mcModel   = mcFac.mcFactory().createMcModel( self.evoOptions )
        
        # NOTE: if we need to update parameters, we should update this copy.
        self.params    = self.mcModel.params

        # --------------------------------------------------------------------------
        # initialize empty arrays to track evolution
        # --------------------------------------------------------------------------
        self.tmax       = simArryData['tmax']           # max number of iterations to simulate
        self.tcap       = simArryData['tcap']           # num of iterations between each stats check

        self.nij        = simArryData['nij']            # 2d array for abundances
        self.bij_mutCnt = simArryData['bij_mutCnt']     # 2d array for b mutation counts    
        self.dij_mutCnt = simArryData['dij_mutCnt']     # 2d array for d mutation counts
        self.cij_mutCnt = simArryData['cij_mutCnt']     # 2d array for c mutation counts
        
        # NOTE1: we have three parameter sets in this class
        #
        # 1. self.paramsself.mcModel.params
        #      - used to store parameter changes to avoid having to update input files
        #
        # 2. self.evoOptions.params
        #      - initially read from the input file provided, but can be updated
        #        with changes to self.params using the method recaculate_mcModel()
        #
        # 3. self.mcModel.params
        #      - initially generate from self.evoOptions.params to generate an 
        #        an MC model. This set is updated when recaculate_mcModel() is 
        #        called. These parameters reflect those from evoOptions used to
        #        generate the MC model.
        #
        # NOTE2: parameter updates are only required when a model needsd to be
        #        generated after updating params.
        #
        # NOTE3: The MC model should not be updated if changes to self.params
        #        for altering sim dynamics, but not state space, should not be
        #        passed to self.mcModel.params and self.evoOptions.params using
        #        recaculate_mcModel(). For example
        #            Ua, Uc = 0 to turn off mutations
        #            se, R  = 0 to turn off environmental changes
        #        These can be set directly to alter sim dynamics alone, but not
        #        the state space.
    
    # --------------------------------------------------------------------------
    
    def get_simPathsIO_dict(self):
        # get_simPathsIO_dict is used to retrieve IO dictionary that was provided to 
        # instantiate of object of type SimEvoInit
        simIODict_out = dict()

        simIODict_out['paramFile'] = self.paramFile # parameter file for MC model
        simIODict_out['paramTag'] = self.paramTag   # param file tag to idetify input for outputs

        # output paths for data
        simIODict_out['simDatDir'] = self.simDatDir   # directory for outputs
        simIODict_out['statsFile'] = self.simDatFile1 # file name for stats print outs
        simIODict_out['snpshtFile'] = self.simDatFile2 # file name for end-of-run snapshot
        
        simIODict_out['modelDynamics']      = self.modelDynamics
        simIODict_out['simpleEnvShift']     = self.simpleEnvShift
        simIODict_out['modelType']          = self.modelType
        simIODict_out['absFitType']         = self.absFitType

        return simIODict_out
    
    # --------------------------------------------------------------------------
    
    def get_simArryData_dict(self):
        # get_simArryData_dict is used to retrieve array data dictionary that was provided to 
        # instantiate of object of type SimEvoInit
        simDataDict_out = dict()

        simDataDict_out['tmax'] = self.tmax # max number of iterations to simulate
        simDataDict_out['tcap'] = self.tcap # num of iterations between each stats check
        simDataDict_out['nij']  = self.nij  # 2d array for abundances

        simDataDict_out['bij_mutCnt'] = self.bij_mutCnt # 2d array for b mutation counts    
        simDataDict_out['dij_mutCnt'] = self.dij_mutCnt # 2d array for d mutation counts
        simDataDict_out['cij_mutCnt'] = self.cij_mutCnt # 2d array for c mutation counts

        return simDataDict_out
    
    # --------------------------------------------------------------------------
    
    def recaculate_mcModel(self):
        # recaculate_mcModel() allows you to update the associated mcModel after
        # parameters have been updated. This prevents the need to update paramter
        # files directly.
        
        if self.check_newParams():
            # tranfer the updated parameter set to the evoOptions copy
            self.evoOptions.params = self.params
            
            # regenerate the mcModel. 
            # Note: user is responsible for using valid paramters. There are not che
            self.mcModel   = mcFac.mcFactory().createMcModel( self.evoOptions )
        
        return None
    
    # --------------------------------------------------------------------------
    
    def check_newParams(self):
        # simple function to prevent errors in computing an MC model that isn't 
        # valid. For now we assume the b-evo DRE model
        
        validParams = True
        paramErr = []
        
        # NOTE: check below is an implementation for bEvo evolution with DRE.
        #       conditions for other models have not been implemented
        
        # Following parameters cannot be zero. 
        validParams = validParams and (self.params['Ua'] > 0 )
        if not validParams:
            paramErr.append('Ua')
        validParams = validParams and (self.params['Uc'] > 0 )
        if not validParams:
            paramErr.append('Uc')
        validParams = validParams and (self.params['cp'] > 0 )
        if not validParams:
            paramErr.append('cp')
        validParams = validParams and (self.params['sa_0'] > 0 )
        if not validParams:
            paramErr.append('sa_0')
        validParams = validParams and (self.params['d'] > 1 )
        if not validParams:
            paramErr.append('d')
        validParams = validParams and (self.params['T'] > 1e3 )
        if not validParams:
            paramErr.append('T')
        validParams = validParams and (self.params['alpha'] <= 1.0 )
        if not validParams:
            paramErr.append('alpha')
        
        if not validParams:
            errMsg = "Invalid Inputs for: " + ("%s,"*len(paramErr))
            print(errMsg % tuple(paramErr))
        
        return validParams
    
    # --------------------------------------------------------------------------
    
    def turn_off_mutations(self,mutList=None):
        # simple method to turn off mutations in simulation
        # this will not update the MC model
        try:
            if (mutList == None):
                # update mutation rates to turn off mutations
                self.params['Ua'] = 0
                self.params['Uc'] = 0
                
                self.params['UaDel'] = 0
                self.params['UcDel'] = 0
            else:
                for mutType in mutList:
                    self.params[mutType] = 0
        except:
            print('invalid list of mutation paramters')
        return None
    
    # --------------------------------------------------------------------------
    
    def turn_off_environmentalChanges(self):
        # simple method to rate of environmental changes in simulation
        # this will not update the MC model
        
        # update mutation rates to turn off mutations
        self.params['se'] = 0
        
        return None
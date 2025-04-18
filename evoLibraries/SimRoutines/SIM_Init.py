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
        self.modelDynamics = simPathsIO['modelDynamics']
        self.simpleEnvShift     = simPathsIO['simpleEnvShift']
        self.modelType  = simPathsIO['modelType']
        self.absFitType = simPathsIO['absFitType']

        # generate the MC model 
        #   note: parameters are capture as a member of this class
        #   note: set pfixSolver type to 3 (use selection coeff) for faster calculations
        #         since we don't actually need pfix for the simulations
        tempEvoOptions =  evoObj.evoOptions(self.paramFilePath,self.modelType,self.absFitType)
        tempEvoOptions.params['pfixSolver'] = 3
        self.mcModel   = mcFac.mcFactory().createMcModel( tempEvoOptions )

        # --------------------------------------------------------------------------
        # initialize empty arrays to track evolution
        # --------------------------------------------------------------------------
        self.tmax       = simArryData['tmax']           # max number of iterations to simulate
        self.tcap       = simArryData['tcap']           # num of iterations between each stats check

        self.nij        = simArryData['nij']            # 2d array for abundances
        self.bij_mutCnt = simArryData['bij_mutCnt']     # 2d array for b mutation counts    
        self.dij_mutCnt = simArryData['dij_mutCnt']     # 2d array for d mutation counts
        self.cij_mutCnt = simArryData['cij_mutCnt']     # 2d array for c mutation counts
        
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
   
    
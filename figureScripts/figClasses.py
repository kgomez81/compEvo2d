# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:09:32 2024

@author: Owner
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle 

import os
import sys

from evoLibraries.MarkovChain import MC_array_class as mcArry
import figFunctions as figFun

#%% Main class for rho plots

class figRhoPlot():
    # class to automate the process of generating data for rho figures.
    
    #% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,figPathsIO,figModelIO):
        # --------------------------------------------------------------------------
        # fields to specify input and output paths
        # --------------------------------------------------------------------------
        self.inputsPath  = os.path.join(os.getcwd(),'inputs')
        self.outputsPath = os.path.join(os.getcwd(),'outputs')
        self.figSavePath = os.path.join(os.getcwd(),'figures',figPathsIO['saveFigSubdir'])
        
        self.figFile     = figPathsIO['figFile']
        self.figDatDir   = figPathsIO['figDatDir']
        self.paramFile   = figPathsIO['paramFile']
        self.paramTag    = figPathsIO['paramTag']
        self.saveDatFile = ''.join(('_'.join((self.figDatDir,self.paramTag)),'.pickle'))

        # set paths to generate output files for tracking progress of loop/parloop
        self.mcArrayOutputPath   = os.path.join(self.outputsPath,self.figDatDir) 
        self.saveDatFilePath     = os.path.join(self.mcArrayOutputPath,self.saveDatFile)
        self.figFilePath         = os.path.join(self.figSavePath,self.figFile)

        # The parameter file is read and a dictionary with their values is generated.
        self.paramFilePath = os.path.join(self.inputsPath,self.paramFile)
        
        # --------------------------------------------------------------------------
        # fields to specify model, as well as to calculate, retrieve and store data
        # --------------------------------------------------------------------------
        self.modelType  = figModelIO['modelType']
        self.absFitType = figModelIO['absFitType']
        
        # set list of variable names that will be used to specify the grid
        # and the bounds with increments needed to define the grid.
        # varNames[0] = string with dictionary name of evo model parameter
        # varNames[1] = string with dictionary name of evo model parameter
        #
        # varBounds values define the min and max bounds of parameters that are used to 
        # define the square grid. First index j=0,1 (one for each evo parameter). 
        # varBounds[0]    = list of base 10 exponentials to use in forming the parameter 
        #                   grid for X1
        # varBounds[1]    = list of base 10 exponentials to use in forming the parameter 
        #                   grid for X2
        # NOTE: both list should include 0 to represent the center points of the grid.
        #       For example, [-2,-1,0,1,2] would designate [1E-2,1E-1,1E0,1E1,1e2].
        #       Also note that the entries don't have to be integers.
        self.varNames   = figModelIO['varNames']
        self.varBounds  = figModelIO['varBounds']
        self.nArry      = figModelIO['nArry']
        self.mcModels   = []
        self.get_McArrayData()
        
    #% ------------------------------------------------------------------------
    # generate MC data
    # --------------------------------------------------------------------------
    
    def get_McArrayData(self):
        # get the mcArray data
        if not (os.path.exists(self.mcArrayOutputPath)):
            # if the data does not exist then generate it
            os.mkdir(self.mcArrayOutputPath)
            
            # generate grid
            tic = time.time()
            self.mcModels = mcArry.mcEvoGrid(self.paramFilePath, 
                                             self.modelType, 
                                             self.absFitType, 
                                             self.varNames, 
                                             self.varBounds, 
                                             self.mcArrayOutputPath)
            print(time.time()-tic)
            
            # save the data to a pickle file
            outputs  = [self.paramFilePath, self.modelType, self.absFitType, self.varNames, self.varBounds, self.mcModels]
            with open(self.saveDatFilePath, 'wb') as file:
                # Serialize and write the variable to the file
                pickle.dump(outputs, file)
        
        else:
            print('Warning: Data for Markov Chain Models was found in the output directory. Data will be loaded from the existing output files.')
            # if data exist, then just load it to generate the figure
            with open(self.saveDatFilePath, 'rb') as file:
                # Serialize and write the variable to the file
                loaded_data = pickle.load(file)
                
            self.mcModels        = loaded_data[5]

    def get_rhoPlotData(self):

        # ------------------------------------------------------------------------
        # construct plot variables
        # --------------------------------------------------------------------------
        self.log10_ss   = np.log10(self.mcModels.eff_sc_ij / self.mcModels.eff_sa_ij)   # sc/sd
        self.log10_UU   = np.log10(self.mcModels.eff_Ua_ij / self.mcModels.eff_Uc_ij)   # Ud/Uc
        self.log10_rho  = np.log10(self.mcModels.rho_ij)                           # rho
        
        
        [x,y,z] = figFun.getScatterData_special(np.log10(self.mcModels.eff_sc_ij / self.mcModels.eff_sa_ij),
                                                np.log10(self.mcModels.eff_Ua_ij / self.mcModels.eff_Uc_ij),
                                                np.log10(self.mcModels.rho_ij))
        
        # plot bounds
        xBnds = [int(np.floor(min(x))), int(np.ceil(max(x))+1)]
        yBnds = [int(np.floor(min(y))), int(np.ceil(max(y))+1)]
        zBnds = [-0.15                , 0.15                  ]
        
        # plot ticks
        xTicks      = [-1,-0.5,0,0.5,1]
        yTicks      = [-2,-1,0,1,2]
        zTicks      = [np.round(ii*0.05,2) for ii in range(-3, 4)]
        
        # plot labels
        xTickLbls   = [str(0.1),'',str(1),'',str(10)]
        yTickLbls   = [str(0.01),str(0.1),str(1),str(10),str(100)]
        zTickLbls   = [("%.2f" % 10**tick) for tick in zTicks]
        
        figData = {'xData': x,
                   'xBnds': xBnds,
                   'xTick': xTicks,
                   'xLbls': xTickLbls,
                   'yData': y,
                   'yBnds': yBnds,
                   'yTick': yTicks,
                   'yLbls': yTickLbls,
                   'zData': z,
                   'zBnds': zBnds,
                   'zTick': zTicks,
                   'zLbls': zTickLbls,
                   'log10_ss':  self.log10_ss,
                   'log10_UU':  self.log10_UU,
                   'log10_rho': self.log10_rho,
                   }
        
        return figData        
    
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description:
Defines the basics functions used in all scripts that process matlab
data and create figures in the mutation-driven adaptation manuscript.
"""
# *****************************************************************************
# libraries
# *****************************************************************************

import numpy as np
import csv

# *****************************************************************************
#                       my classes and structures
# *****************************************************************************

class evoOptions:
    # evoOptions encapsulates all evolution parameters needed to build a 
    # Markov Chain model.
    
    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    
    def __init__(self,paramFilePath,modelType):
        # file path to csv with parameters
        self.paramFilePath  = paramFilePath
        
        # set model type (str = 'RM' or 'DRE') 
        self.modelType      = modelType
        
        # read the paremeter files and store as dictionary
        self.options_readParameterFile()
        
    # --------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------
    
    def options_readParameterFile(self):
        # This function reads the parameter values a file and creates a dictionary
        # with the parameters and their values 
    
        # create array to store values and define the parameter names
        if self.modelType == 'RM':
            paramList = ['T','b','dOpt','sd','UdMax','UdDel','cp','Uc','UcDel','R','se']
        else:
            paramList = ['T','b','dOpt','alpha','Ud','UdDel','cp','Uc','UcDel','R','se','jStart','cdfOption']
        paramValue = np.zeros([len(paramList),1])
        
        # read values from csv file
        with open(self.paramFilePath,'r') as csvfile:
            parInput = csv.reader(csvfile)
            for (line,row) in enumerate(parInput):
                paramValue[line] = float(row[0])
        
        # create dictionary with values
        self.params = dict([[paramList[i],paramValue[i][0]] for i in range(len(paramValue))])
        
        return None

# --------------------------------------

class evoGridOptions(evoOptions):
    # evoGridOptions encapsulates evolution parameters and bounds to define 
    # a grid of Markov Chain models for figures.
    
    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    
    def __init__(self,paramFilePath,modelType,saveDataName,saveFigName,varNames,varBounds):
        
        super().__init__(paramFilePath,modelType)
        
        # save path for array with MC sampling 
        self.saveDataName   = saveDataName
        
        # set list of variable names that will be used to specify the grid
        # and the bounds with increments needed to define the grid.
        
        # square array is built with first two parmeters, and second set are held constant.
        # varNames[0][0] stored as X1_ARRY
        # varNames[1][0] stored as X1_ref
        # varNames[0][1] stored as X2_ARRY
        # varNames[1][1] stored as X2_ref
        self.varNames       = varNames
        
        # varBounds values define the min and max bounds of parameters that are used to 
        # define the square grid. 
        # varBounds[j][0] = min Multiple of parameter value in file (Xj variable)
        # varBounds[j][1] = max Multiple of parameter value in file (Xj variable)
        # varBounds[j][2] = number of increments from min to max (log scale) 
        self.varBounds      = varBounds
        
    # --------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------
    
    def get_params_ij(ii,jj):
        # get_params_ij set the center param list for 
        
        return params_ij

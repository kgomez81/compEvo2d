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
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    
    def __init__(self,paramFilePath,modelType,absFitType):
        # file path to csv with parameters
        self.paramFilePath  = paramFilePath
        
        # set model type (str = 'RM' or 'DRE') 
        self.modelType      = modelType
        
        # set absolute fitness type (str = 'RM' or 'DRE') 
        self.absFitType     = absFitType
        
        # initialize with empty params dictionary
        self.params = {}
        
        # initialize with empty params dictionary
        self.paramsList = []
        
        # read the paremeter files and store as dictionary
        self.options_readParameterFile()
        
    #%% ------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------
    
    def options_readParameterFile(self):
        # This function reads the parameter values a file and creates a dictionary
        # with the parameters and their values 
    
        # create array to store values and define the parameter names
        # absolute fitness parameters designated with "a" in variable name
        if (self.modelType == 'RM') and (self.absFitType == 'dEvo'):
            
            self.paramList = ['T','b','dOpt','sa','UaMax','UaDel','cp','Uc','UcDel','R','se','pfixSolver','parallelSelect']
            
        elif (self.modelType == 'RM') and (self.absFitType == 'bEvo'):
            
            self.paramList = ['T','d','bMax','sa','UaMax','UaDel','cp','Uc','UcDel','R','se','pfixSolver','parallelSelect']
            
        elif (self.modelType == 'DRE') and (self.absFitType == 'dEvo'):
            
            self.paramList = ['T','b','dOpt','alpha','Ua','UaDel','cp','Uc','UcDel','R','se','iMax','jStart','cdfOption','pfixSolver','parallelSelect']
        
        elif (self.modelType == 'DRE') and (self.absFitType == 'bEvo'):
            
            self.paramList = ['T','d','bMax','sa_0','Ua','UaDel','cp','Uc','UcDel','R','se','iMax','alpha','DreMod','pfixSolver','parallelSelect'] 
        
        # save the number of parameters
        nParams = len(self.paramList)
        
        # generate list to store values from csv file
        paramValue = np.zeros([nParams,1])
        
        # read values from csv file
        with open(self.paramFilePath,'r') as csvfile:
            parInput = csv.reader(csvfile)
            for (line,row) in enumerate(parInput):
                paramValue[line] = float(row[0])
        
        # create dictionary with values
        self.params = dict([[self.paramList[i],paramValue[i][0]] for i in range(nParams)])
        
        return None

    # --------------------------------------------------------------------------

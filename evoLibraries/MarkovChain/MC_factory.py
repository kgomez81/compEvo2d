# -*- coding: utf-8 -*-
"""
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description: MC Factory class for generating RM and DRE Markov Chain models
approximating evolution with the Bertram & Masel 2019 variable density lottery model.
"""

from evoLibraries import evoObjects as evoObj

from evoLibraries.MarkovChain import MC_DRE_class as mcDRE
from evoLibraries.MarkovChain import MC_RM_class as mcRM

class mcFactory():
    
    def __init__(self):
        pass
        
    def newMcModel(self,paramFilePath,modelType,absFitType):
        # Get parameters for model
        mcParams = evoObj.evoOptions(paramFilePath, modelType, absFitType)
        return self.createMcModel(mcParams)
        
    def createMcModel(self,mcParams):
        # Build the MC model
        if   (mcParams.modelType == 'RM' ):
            return mcRM.mcEvoModel_RM(mcParams)    
        
        elif (mcParams.modelType == 'DRE'):
            return mcDRE.mcEvoModel_DRE(mcParams)
        
        else:
            print('Error: Model type is not supported.')    
            return None
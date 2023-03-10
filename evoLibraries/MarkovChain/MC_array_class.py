# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description: Defintion of a class that defines an array of Markov Chain models
for evolution that approximate the evolution in the Bertram & Masel 2019 
variable density lottery model.
"""


# *****************************************************************************
# import libraries
# *****************************************************************************

import numpy as np


from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

class mcEvoGrid(evoObj.evoOptions):
    # evoGridOptions encapsulates evolution parameters and bounds to define 
    # a grid of Markov Chain models for figures.
    
    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    
    def __init__(self,paramFilePath,modelType,saveDataName,saveFigName,varNames,varBounds):
        
        # intialize member variables that are part of evoOptions super class
        super().__init__(paramFilePath,modelType)
        
        # save path for array with MC sampling 
        self.saveDataName   = saveDataName
        
        # set list of variable names that will be used to specify the grid
        # and the bounds with increments needed to define the grid.
        # varNames[0] = string with dictionary name of evo model parameter
        # varNames[1] = string with dictionary name of evo model parameter
        self.varNames       = varNames
        
        # varBounds values define the min and max bounds of parameters that are used to 
        # define the square grid. First index j=0,1 (one for each evo parameter). 
        # varBounds[0]    = list of base 10 exponentials to use in forming the parameter 
        #                   grid for X1
        # varBounds[1]    = list of base 10 exponentials to use in forming the parameter 
        #                   grid for X2
        # NOTE: both list should include 0 to represent the center points of the grid.
        #       For example, [-2,-1,0,1,2] would designate [1E-2,1E-1,1E0,1E1,1e2].
        #       Also note that the entries don't have to be integers.
        self.varBounds      = varBounds

        # define arrays to save all effective evo parameters, i.e. evo parameters
        # at the intersection.
        self.intersect_state_ij = np.zeros(self.get_evoArray_dim())
        
        # density and absolute fitness at intersection
        self.eff_eff_y_ij   = np.zeros(self.get_evoArray_dim())        
        self.eff_eff_d_ij   = np.zeros(self.get_evoArray_dim())        
        
        # arrays with effective evolution parameters at intersections
        self.eff_N_ij       = np.zeros(self.get_evoArray_dim())
        self.eff_Ud_ij      = np.zeros(self.get_evoArray_dim())
        self.eff_Uc_ij      = np.zeros(self.get_evoArray_dim())
        self.eff_sd_ij      = np.zeros(self.get_evoArray_dim())
        self.eff_sc_ij      = np.zeros(self.get_evoArray_dim())
        self.eff_pFix_d_ij  = np.zeros(self.get_evoArray_dim())
        self.eff_pFix_c_ij  = np.zeros(self.get_evoArray_dim())
        
        # arrays with evolution/environment rates
        self.eff_vd_ij  = np.zeros(self.get_evoArray_dim())
        self.eff_vc_ij  = np.zeros(self.get_evoArray_dim())
        self.eff_ve_ij  = np.zeros(self.get_evoArray_dim())
        
        # get the full set of effective evo parameters and rates
        self.get_evoGrid_effEvoParams()
        
    # --------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------
    
    def get_evoArray_dim(self):
        # return a tuple with the dimensions of the grid 
        
        # get the size of the lists. subtract 1 to align size with python 
        evoGridDim = ( len(self.varBounds[0]) , len(self.varBounds[1]) )
        
        return evoGridDim
    
    # --------------------------------------------------------------------------
    
    def get_params_ij(self,ii,jj):
        # get_params_ij set the center param list for 
        
        # check if ii and jj are permissible indices
        evoGridDim = self.get_evoArray_dim
        
        if (ii > evoGridDim[0]-1) or (jj > evoGridDim[1]-1): 
            # if indices invalid, then return empty list
            params_ij = {}
        else:
            # get primary dictionary with evo parameters
            params_ij = self.params
            
            # scale evo parameters that were selected to vary by the appropriate of 10**k
            params_ij[self.varNames[0]] = params_ij[self.varNames[0]] * 10**self.varBounds[0][ii]
            params_ij[self.varNames[1]] = params_ij[self.varNames[1]] * 10**self.varBounds[1][jj]
        
        return params_ij
    
    # --------------------------------------------------------------------------
    
    def get_evoGrid_effEvoParams(self):
        # get_evoGrid_effEvoParams() runs through all possible MC models with 
        # parameters from the grid and calculates the effective evo parameters
        # (i.e. the evo params at intersection)
        
        evoGridDim = self.get_evoArray_dim()
        
        # loop through each MC model
        for ii in range(evoGridDim[0]):
            for jj in range(evoGridDim[1]):
                
                # check the MC model type and get intersection evo params
                if (self.modelType == 'RM'):
                    # get the MC evo model and find the intersection
                    temp_mcModel = mcRM.mcEvoModel_RM( self.get_params(ii,jj) )
                else:
                    # get the MC evo model and find the intersection
                    temp_mcModel = mcRM.mcEvoModel_DRE( self.get_params(ii,jj) )
                    
                # calculate intersections and find the stochastically stable
                # state of absolute fitness
                mc_stable_state = temp_mcModel.get_mc_stable_state()
                
                # save all evo parameter values
                self.intersect_state_ij[ii,jj] = mc_stable_state
                
                # density and absolute fitness at intersection
                self.eff_eff_y_ij[ii,jj]   = temp_mcModel.eq_yi[mc_stable_state]
                self.eff_eff_d_ij[ii,jj]   = temp_mcModel.di[mc_stable_state]
                
                # arrays with effective evolution parameters at intersections
                self.eff_N_ij[ii,jj]       = temp_mcModel.eq_Ni[mc_stable_state]
                self.eff_Ud_ij[ii,jj]      = temp_mcModel.Ud_i[mc_stable_state]
                self.eff_Uc_ij[ii,jj]      = temp_mcModel.Uc_i[mc_stable_state]
                self.eff_sd_ij[ii,jj]      = temp_mcModel.sd_i[mc_stable_state]
                self.eff_sc_ij[ii,jj]      = temp_mcModel.sc_i[mc_stable_state]
                self.eff_pFix_d_ij[ii,jj]  = temp_mcModel.pFix_d_i[mc_stable_state]
                self.eff_pFix_c_ij[ii,jj]  = temp_mcModel.pFix_c_i[mc_stable_state]
                
                # arrays with evolution/environment rates
                self.eff_vd_ij[ii,jj]  = temp_mcModel.vd_i[mc_stable_state]
                self.eff_vc_ij[ii,jj]  = temp_mcModel.vc_i[mc_stable_state]
                self.eff_ve_ij[ii,jj]  = temp_mcModel.ve_i[mc_stable_state]
                
        
        return None
    
    # --------------------------------------------------------------------------
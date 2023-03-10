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
from evoLibraries.MarkovChain import MC_functions as mcFun

class mcEvoGrid(evoObj.evoOptions):
    # evoGridOptions encapsulates evolution parameters and bounds to define 
    # a grid of Markov Chain models for figures.
    
    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    
    def __init__(self,paramFilePath,modelType,varNames,varBounds):
        
        # intialize member variables that are part of evoOptions super class
        super().__init__(paramFilePath,modelType)
        
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
        self.eff_y_ij   = np.zeros(self.get_evoArray_dim())        
        self.eff_d_ij   = np.zeros(self.get_evoArray_dim())        
        
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
        
        # arrays for vd and vc regime ids
        self.eff_evoRegime_d_ij  = np.zeros(self.get_evoArray_dim())
        self.eff_evoRegime_c_ij  = np.zeros(self.get_evoArray_dim())
        
        # density and absolute fitness at intersection
        self.rho_ij     = np.zeros(self.get_evoArray_dim())        
        
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
        evoGridDim = self.get_evoArray_dim()
        
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
    
    def get_evoParam_ij(self,ii,jj,evoParamName,axisNum):
        # get_evoParam_ij calculates the specif evo param for the ii,jj element
        # of the MC evo grid. modelNum indicates the axis to use, i.e the x-axis
        # is model 1 that varies the parameter varNames[0], and the y-axis is 
        # model 2 that varies the parameter varnames[1]
        
        # check if ii and jj are permissible indices
        evoGridDim = self.get_evoArray_dim()
        
        if (ii > evoGridDim[0]-1) or (jj > evoGridDim[1]-1) or ( not self.params.has_key(evoParamName) ): 
            # if indices invalid, then return empty list
            evoParam_ij = []
            
        elif (axisNum == 1) and (evoParamName == self.varNames[0]):
            # get value primary dictionary and calculate value after scaling by
            # power of 10 in varBounds
            evoParam_ij = self.params[evoParamName] * 10**self.varBounds[0][ii]
            
        elif (axisNum == 2) and (evoParamName == self.varNames[1]):
            # get value primary dictionary and calculate value after scaling by
            # power of 10 in varBounds
            evoParam_ij = self.params[evoParamName] * 10**self.varBounds[1][jj]
        else:
            # if requesting parameter that doesn't vary in model, then just
            # return the base parameter value
            evoParam_ij = self.params[evoParamName]
                
        return evoParam_ij
    
    # --------------------------------------------------------------------------
    
    def get_evoParam_grid(self,evoParamName,axisNum):
        # get_evoParam_grid calculates the full grid of evo param values used 
        # to make the MC evo grid.
        
        # build grid to return values
        evoGridDim = self.get_evoArray_dim()
        evoParam_grid = np.zeros(evoGridDim)
        
        for ii in range(evoGridDim[0]):
            for jj in range(evoGridDim[1]):
                # load the evo param values at each entry of the evo grid
                evoParam_grid[ii,jj] = self.get_evoParam_ij(ii,jj,evoParamName,axisNum)
        
        return evoParam_grid
    
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
                    temp_mcModel = mcRM.mcEvoModel_RM( self.get_params_ij(ii,jj) )
                else:
                    # get the MC evo model and find the intersection
                    temp_mcModel = mcRM.mcEvoModel_DRE( self.get_params_ij(ii,jj) )
                    
                # calculate intersections and find the stochastically stable
                # state of absolute fitness
                mc_stable_state = temp_mcModel.get_mc_stable_state()
                
                # save all evo parameter values
                self.intersect_state_ij[ii,jj] = mc_stable_state
                
                # density and absolute fitness at intersection
                self.eff_y_ij[ii,jj]   = temp_mcModel.eq_yi[mc_stable_state]
                self.eff_d_ij[ii,jj]   = temp_mcModel.di[mc_stable_state]
                
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
                
                # save evo regimes
                self.eff_evoRegime_d_ij[ii,jj]  = temp_mcModel.evoRegime_d_i[mc_stable_state]
                self.eff_evoRegime_c_ij[ii,jj]  = temp_mcModel.evoRegime_c_i[mc_stable_state]
                
                # calculate rho of the MC model
                self.rho_ij[ii,jj]  = temp_mcModel.calculate_evoRho()
                
        return None
    
    # --------------------------------------------------------------------------
    
    
    
    # --------------------------------------------------------------------------
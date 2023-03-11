# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description: Defintion of abstract MC class for defining Markov Chain models
of evolution that approximate the evolution in the Bertram & Masel 2019 
variable density lottery model.
"""

# *****************************************************************************
# import libraries
# *****************************************************************************

import numpy as np
import csv

from abc import ABC, abstractmethod

from evoLibraries.MarkovChain import MC_functions as mcFun

class mcEvoModel(ABC):
    # ABSTRACT class used to prototype the MC classes for RM and DRE
    #
    # IMPORTANT: This class is not mean to be used for anything other then for 
    # consolidating the common code of RM and DRE MC classes, i.e. something like 
    # an Abstract class
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,params):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        self.params = params            # dictionary with evo parameters
    
        # absolute fitness landscape (array of di terms), 
        self.di = np.array([0]) 
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.shape) # state number
        self.Ud_i    = np.zeros(self.di.shape) # absolute fitness mutation rate
        self.Uc_i    = np.zeros(self.di.shape) # relative fitness mutation rate
        self.eq_yi   = np.zeros(self.di.shape) # equilibrium density of fitness class i
        self.eq_Ni   = np.zeros(self.di.shape) # equilibrium population size of fitness class i
        self.sd_i    = np.zeros(self.di.shape) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.shape) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_d_i = np.zeros(self.di.shape) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.vd_i    = np.zeros(self.di.shape) # rate of adaptation in absolute fitness trait alone
        self.vc_i    = np.zeros(self.di.shape) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.shape) # rate of fitness decrease due to environmental degradation
        
        self.evoRegime_d_i = np.zeros(self.di.shape) # regim ID identifying Successional, Mult Mut, Diffusion
        self.evoRegime_c_i = np.zeros(self.di.shape) # regim ID identifying Successional, Mult Mut, Diffusion
        
    #%%----------------------------------------------------------------------------
    # abstract methods
    #------------------------------------------------------------------------------
    
    @abstractmethod
    def get_absoluteFitnessClasses(self):
        "Method that defines the set of di terms (absolute fitness landscape)"
        "for the model given the evolution parameters that have been provided"
        pass
    
    #------------------------------------------------------------------------------
    
    @abstractmethod
    def get_stateSpaceEvoParameters(self):
        "Method that defines the arrays for evolution parameters at each state"
        pass
    
    #------------------------------------------------------------------------------
    
    @abstractmethod
    def get_stateSpacePfixValues(self):
        "Method that defines the arrays for pfix values at each state"
        pass
        
    #------------------------------------------------------------------------------
    
    @abstractmethod
    def get_stateSpaceEvoRates(self):
        "Method that defines the arrays for evolution rates at each state"
        pass
    
    #------------------------------------------------------------------------------
    
    @abstractmethod
    def get_v_intersect_state_index(self,v2):      
        "get_v_intersect_state() returns the intersection state of v2 evo rate array  "
        "with vd. Implementation of method varies for RM & DRE inheriting classes. RM "
        "orders states from most beneficial to least (index-wise), and DRE is reversed" 
        "                                                                             "
        " NOTE: vd in comparison to v2 because it is v2's relationship with vd that   "
        "       we use to determine if we return the extinction class dExt, or the    "
        "       highest fitness class in the vd_i array.                              "
        pass
    
    #%%----------------------------------------------------------------------------
    # Concrete methods (common to both RM and DR MC class implementations)
    #------------------------------------------------------------------------------
    
    def get_vd_i_perUnitTime(self):      
        # get_vd_i_perUnitTime()  returns the set of vd_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        # NOTE: vd saved in time-scale of wild type generations
        
        vd_i_perUnitTime = np.asarray([ self.vd_i[ii]*(self.di[ii]-1) for ii in range(self.di.size)])
        
        return vd_i_perUnitTime
    
    #------------------------------------------------------------------------------
    
    def get_vc_i_perUnitTime(self):      
        # get_vc_perUnitTime()  returns the set of vc_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        
        vc_i_perUnitTime = np.asarray([ self.vc_i[ii]*(self.di[ii]-1) for ii in range(self.di.size)])
        
        return vc_i_perUnitTime
    
    #------------------------------------------------------------------------------
    
    def get_ve_i_perUnitTime(self):      
        # get_ve_i_perUnitTime()  returns the set of ve_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        
        ve_i_perUnitTime = np.asarray([ self.ve_i[ii]*(self.di[ii]-1) for ii in range(self.di.size)])
        
        return ve_i_perUnitTime
    
    #------------------------------------------------------------------------------

    def get_vd_ve_intersection_index(self):      
        # get_vd_ve_intersection() returns the state for which vd and ve are closest.
        # Serves as a wrapper for generic method get_v_intersect_state_index()
        
        # NOTE: we are calculating the index of the stable state, not the stable state
        iStableState_index = self.get_v_intersect_state_index(self.vd_i, self.ve_i)
        
        return iStableState_index
    
    #------------------------------------------------------------------------------
    
    def get_vd_vc_intersection_index(self):      
        # get_vd_ve_intersection() returns the state for which vd and vc are closest
        # Serves as a wrapper for generic method get_v_intersect_state_index()
        
        # NOTE: we are calculating the index of the stable state, not the stable state
        iStableState_index = self.get_v_intersect_state_index(self.vd_i, self.vc_i)        

        return iStableState_index
    
    #------------------------------------------------------------------------------
    
    def get_mc_stable_state(self):      
        # get_mc_stable_state() returns the MC stochastically stable absolute
        # fitness state. This will be whatever v-intersection is reached first
        # from the extinction state.
        
        # calculate intersection states (SS = Stable State)
        idx_SS = [self.get_vd_ve_intersection(), self.get_vd_vc_intersection()]
        
        # find the intersection state closest to extinction, which requires taking
        # taking max of the two intersection states.
        mc_stable_state = np.argmin( idx_SS )
        
        return mc_stable_state
    
    #------------------------------------------------------------------------------
    
    def calculate_evoRho(self):
        # This function calculate the rho parameter defined in the manuscript,
        # which measures the relative changes in evolution rates due to increases
        # in max available territory parameter
        
        # CONSTRUCT DICTIONARY OF EVO PARAMS TO GET RHO
        # 's'     - selection coefficient
        # 'pFix'  - probability of fixation 
        # 'U'     - beneficial mutation rate
        # 'regID' - regime ID for rate of adaptation
        #           List of possible regime IDs 
        #            0: Bad evo parameters (either U, s, N, pFix == 0)
        #            1: successional
        #            2: multiple mutations
        #            3: diffusion
        #           -1: regime undetermined, i.e. in transition region   
        stable_state = self.get_mc_stable_state()
        
        # build evo param dictionary for absolute fitness trait
        evoParams_absTrait = {'s'     : self.sd_i[stable_state], \
                              'pFix'  : self.pFix_d_i[stable_state], \
                              'U'     : self.Ud_i[stable_state], \
                              'regID' : self.evoRegime_d_i[stable_state]}
        
        # build evo param dictionary for relative fitness trait
        evoParams_relTrait = {'s'     : self.sc_i[stable_state], \
                              'pFix'  : self.pFix_c_i[stable_state], \
                              'U'     : self.Uc_i[stable_state], \
                              'regID' : self.evoRegime_c_i[stable_state]}
        
        rho = mcFun.calculate_evoRates_rho(self.eq_Ni[stable_state], \
                                     evoParams_absTrait, evoParams_relTrait)
                
        return rho
        
    #------------------------------------------------------------------------------
    
    def read_pFixOutputs(self,readFile,nStates):
        #
        # CURRENTLY NOT BEING USED. REQUIRES SETTING UP SIMULATION CODE TO ENSURE
        # DATA IS AVILABLE IN FORMAT NEEDED.
        #
        # read_pFixOutputs reads the output file containing estimated pfix values 
        # from simulations and stores them in an array so that they can be used in 
        # creating figures.
        #
        # pFix values for absolute fitness classes should specified beginning from
        # the optimal class, all the way up to the extinction class or greater. The
        # code below will only read up to the extinction class.
        #
        # Inputs:
        # readFile - name of csv file that has the estimated pFix values.
        #
        # Outputs:
        # pFixValues - set of pFix values, beginning from the optimal absolute up to
        #               the extinction class
        
        # create array to store pFix values
        pFixValues = np.zeros([nStates,1])
        
        # read values from csv file
        with open(readFile,'r') as csvfile:
            pfixOutput = csv.reader(csvfile)
            for (line,row) in enumerate(pfixOutput):
                pFixValues[line] = float(row[0])
        
        return pFixValues
        
    #------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description: Defintion of RM MC class for defining Markov Chain models
of evolution that approximate the evolution in the Bertram & Masel 2019 
variable density lottery model.
"""

# *****************************************************************************
# import libraries
# *****************************************************************************

import numpy as np

import evoLibraries.MarkovChain.MC_class as mc
import evoLibraries.MarkovChain.MC_functions as mcFun

import evoLibraries.LotteryModel.LM_functions as lmFun

# *****************************************************************************
# Markov Chain Class - Running Out of Mutations (RM)
# *****************************************************************************

class mcEvoModel_RM(mc.mcEvoModel):
    # class used to encaptulate all of evolution parameters for an Markov Chain (MC)
    # representing a running out of mutations evolution model.
    
    # MC_class for list of class attribures
    #
    # Fitness landscape
    # di      #absolute fitness landscape (array of di terms), 
    
    # state space evolution parameters
    # state_i # state number
    # Ud_i    # absolute fitness mutation rate
    # Uc_i    # relative fitness mutation rate
    # eq_yi   # equilibrium density of fitness class i
    # eq_Ni   # equilibrium population size of fitness class i
    # sd_i    # selection coefficient of "d" trait beneficial mutation
    # sc_i    # selection coefficient of "c" trait beneficial mutation
    
    # state space pFix values
    # pFix_d_i = np.zeros(self.di.shape) # pFix of "d" trait beneficial mutation
    # pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
    
    # state space evolution rates
    # vd_i    # rate of adaptation in absolute fitness trait alone
    # vc_i    # rate of adaptation in relative fitness trait alone
    # ve_i    # rate of fitness decrease due to environmental degradation
    
    #%% ---------------------------------------------------------------------------
    # Class constructor
    # -----------------------------------------------------------------------------
    
    def __init__(self,params):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        super().__init__(params)            # dictionary with evo parameters
        
        # Load absolute fitness landscape (array of di terms)
        self.get_absoluteFitnessClasses() 
        
        # update parameter arrays above
        self.get_stateSpaceEvoParameters()      
        
        # update pFix arrays (expand later to include options)    
        self.get_stateSpacePfixValues()         
        
        # update evolution rate arrays above
        self.get_stateSpaceEvoRates()           
        
    #%%----------------------------------------------------------------------------
    # Definitions for abstract methods
    #------------------------------------------------------------------------------
    
    def get_absoluteFitnessClasses(self):
        # get_absoluteFitnessClasses() generates the sequence of death terms
        # that represent the absolute fitness space. Additionally, with the state
        # space defined, we allocate arrays for all of the other evolutation 
        # parameters and rates.
        
        # Recursively calculate set of absolute fitness classes 
        dMax = self.params['b']+1
        di = [self.params['dOpt']]
        
        while (di[-1] < dMax):
            # loop until at dMax or greater (max death term for viable pop)
            di = di+[di[-1]*(1+self.params['sd']*(di[-1]-1))]
        
        # Finally reverse order and remove dOpt from state space
        self.di = np.asarray(di[::-1][:-1])
        
        # Now that the state space is defined, adjust the size of all other arrays
        # that store evolution parameters and rates
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.size) # state number
        self.Ud_i    = np.zeros(self.di.size) # absolute fitness mutation rate
        self.Uc_i    = np.zeros(self.di.size) # relative fitness mutation rate
        self.eq_yi   = np.zeros(self.di.size) # equilibrium density of fitness class i
        self.eq_Ni   = np.zeros(self.di.size) # equilibrium population size of fitness class i
        self.sd_i    = np.zeros(self.di.size) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.size) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_d_i = np.zeros(self.di.size) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.size) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.vd_i    = np.zeros(self.di.size) # rate of adaptation in absolute fitness trait alone
        self.vc_i    = np.zeros(self.di.size) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.size) # rate of fitness decrease due to environmental degradation
        
        # regime IDs to identify the type of evolution at each state (successional, multi. mutations, diffusion)
        # 0 = bad evo parameters (N,s,U or pFix <= 0)
        # 1 = successional
        # 2 = multiple mutations
        # 3 = diffusion 
        # 4 = regime undetermined
        self.evoRegime_d_i = np.zeros(self.di.shape) 
        self.evoRegime_c_i = np.zeros(self.di.shape) 
        
        return None

    #------------------------------------------------------------------------------
    
    def get_stateSpaceEvoParameters(self):
        
        # calculate population parameters for each of the states in the markov chain model
        # the evolution parameters are calculated along the absolute fitness state space
        # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
        
        yi_option = 3   # numerically solve for equilibrium population densities
        
        # loop through state space to calculate following: 
        # mutation rates, equilb. density, equilb. popsize, selection coefficients
        #
        # NOTE: pFix value not calculate here, but in sepearate function to that method 
        # of getting pFix values can be selected without mucking up the code here.
        for ii in range(self.di.size):
            self.state_i[ii] = -(self.get_iExt()-ii)
            
            # mutation rates (per birth per generation - NEED TO CHECK IF CORRECT)
            self.Ud_i[ii]    = self.params['UdMax']*float(-self.state_i[ii])/self.get_iExt()
            self.Uc_i[ii]    = self.params['Uc']
            
            # population sizes and densities 
            self.eq_yi[ii]   = lmFun.get_eqPopDensity(self.params['b'],self.di[ii],yi_option)
            self.eq_Ni[ii]   = self.params['T']*self.eq_yi[ii]
            
            # selection coefficients ( time scale = 1 generation)
            self.sc_i[ii]    = lmFun.get_c_SelectionCoeff(self.params['b'],self.eq_yi[ii], \
                                                          self.params['cp'],self.di[ii])
            self.sd_i[ii]    = self.params['sd']    # di defined for constant selection coeff
            
        return None 
    
    #%% ----------------------------------------------------------------------------
    #  List of conrete methods from MC class
    # ------------------------------------------------------------------------------

    """
    
    def get_stateSpacePfixValues(self):
        
        # calculate population parameters for each of the states in the markov chain model
        # the evolution parameters are calculated along the absolute fitness state space
        # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
    
    # ------------------------------------------------------------------------------
    
    def get_vd_i_perUnitTime(self):      
        # get_vd_i_perUnitTime()  returns the set of vd_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        # NOTE: vd saved in time-scale of wild type generations
    
    # ------------------------------------------------------------------------------
    
    def get_vc_i_perUnitTime(self):      
        # get_vc_perUnitTime()  returns the set of vc_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
    
    # ------------------------------------------------------------------------------
    
    def get_ve_i_perUnitTime(self):      
        # get_ve_i_perUnitTime()  returns the set of ve_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
    
    # ------------------------------------------------------------------------------
    
    def get_v_intersect_state_index(self,v2):      
        # get_v_intersect_state() returns the intersection state of two evo rate arrays
        # the implementation of this method varies for RM or DRE inheriting classes. RM
        # orders states from most beneficial to least (index-wise), and DRE is reversed
        # v2 should either be self.ve_i or self.vc_i
    
    # ------------------------------------------------------------------------------
    
    def get_vd_ve_intersection_index(self):      
        # get_vd_ve_intersection() returns the state for which vd and ve are closest.
        # Serves as a wrapper for generic method get_v_intersect_state_index()
    
    # ------------------------------------------------------------------------------
    
    def get_vd_vc_intersection_index(self):      
        # get_vd_ve_intersection() returns the state for which vd and vc are closest
        # Serves as a wrapper for generic method get_v_intersect_state_index()
    
    # ------------------------------------------------------------------------------
    
    def get_mc_stable_state(self):      
        # get_mc_stable_state() returns the MC stochastically stable absolute
        # fitness state. This will be whatever v-intersection is reached first
        # from the extinction state.
        
    # ------------------------------------------------------------------------------
    
    def calculate_evoRho(self):                                                           
        # This function calculate the rho parameter defined in the manuscript,            
        # which measures the relative changes in evolution rates due to increases         
        # in max available territory parameter
    
    # ------------------------------------------------------------------------------
    
    def read_pFixOutputs(self,readFile,nStates):                                          
    
         read_pFixOutputs reads the output file containing estimated pfix values
         from simulations and stores them in an array so that they can be used in          
         creating figures.  
                                                                   
    """
    
    #%% ----------------------------------------------------------------------------
    #  Specific methods for the RM MC class
    # ------------------------------------------------------------------------------
    
    def get_iExt(self):      
        # get_iExt()  returns the last state space corresponding to extinction 
        # of the pop. Since dOpt is always zero and not part of the di array, 
        # then dExt is just the size of di array.
        
        return self.di.size
    
    #------------------------------------------------------------------------------
        
    def get_last_di(self):
        # get_last_di() calculates next d-term after di[-1], this value is 
        # occasionally need it to calculate pfix and the rate of adaption.
        
        # get next d-term after last di, which for RM will be dOpt
        di_last = self.params['dOpt']
        
        return di_last
    
    #------------------------------------------------------------------------------
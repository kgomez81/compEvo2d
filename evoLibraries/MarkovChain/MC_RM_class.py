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
        # we also solve for the equilibrium densities along with the absolute 
        # fitness parameters
        yi_option  = 1
        
        if self.absFitType == 'dEvo':
            
            # ###################################### #
            # State space definition for d-Evolution
            # ###################################### #
            
            dMax = self.params['b']+1
            di    = [ self.params['dOpt']                             ]
            bi    = [ self.params['b']                                ]
            eq_yi = [ lmFun.get_eqPopDensity(bi[-1],di[-1],yi_option) ]  # okay to solve here since we start from most fit
            
            while (di[-1] < dMax):
                # loop until at dMax or greater (max death term for viable pop)
                di    = di    + [ di[-1]*(1+self.params['sa']*(di[-1]-1))         ]
                bi    = bi    + [ self.params['b']                                ]
                eq_yi = eq_yi + [ lmFun.get_eqPopDensity(bi[-1],di[-1],yi_option) ]
            
            # Finally reverse order and remove dOpt from state space
            self.di = np.asarray(di[::-1][:-1])
            self.bi = np.asarray(bi[::-1][:-1])
            self.eq_yi = np.asarray(eq_yi[::-1][:-1])
            
        elif self.absFitType == 'bEvo':
            
            # ###################################### #
            # State space definition for b-Evolution
            # ###################################### #

            bi    = [ (self.params['d']-1)*self.params['T'] / (self.params['T']-1) ]
            di    = [ self.params['d']                                             ]
            eq_yi = [ 1/self.params['T']                                           ]  # choose min equil pop density
            
            while (bi[-1] < self.params['bMax']):
                # calculate next b-increment
                # calculate b-increment in steps
                f0 = self.params['sa']*(di[-1]-1)
                f1 = (1-eq_yi[-1]) * np.exp(-bi[-1]*eq_yi[-1])
                
                if ( np.abs( f0 * (1-f1) / f1 ) < 0.0001 ):
                    # small density
                    delta_b = f0 * (bi[-1] + 1)
                else:
                    # large density
                    delta_b = -np.log( 1 - f0 * (1-f1) / f1 ) / eq_yi[-1]
                
                # set next elements of state space
                bi    = bi    + [ bi[-1] + delta_b                                ]
                di    = di    + [ self.params['d']                                ]
                eq_yi = eq_yi + [ lmFun.get_eqPopDensity(bi[-1],di[-1],yi_option) ]  # next equil. pop density
            
            # No need to reverse order here
            self.di    = np.asarray(di)       # b terms for absolute fitness state space
            self.bi    = np.asarray(bi)       # d terms for absolute fitness state space
            self.eq_yi = np.asarray(eq_yi)    # equilibrium density of fitness class i
        
        # Now that the state space is defined, adjust the size of all other arrays
        # that store evolution parameters and rates
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.size) # state number
        self.Ua_i    = np.zeros(self.di.size) # absolute fitness mutation rate
        self.Uc_i    = np.zeros(self.di.size) # relative fitness mutation rate
        self.eq_Ni   = np.zeros(self.di.size) # equilibrium population size of fitness class i
        self.sa_i    = np.zeros(self.di.size) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.size) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_a_i = np.zeros(self.di.size) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.size) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.va_i    = np.zeros(self.di.size) # rate of adaptation in absolute fitness trait alone
        self.vc_i    = np.zeros(self.di.size) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.size) # rate of fitness decrease due to environmental degradation
        
        # regime IDs to identify the type of evolution at each state (successional, multi. mutations, diffusion)
        # 0 = bad evo parameters (N,s,U or pFix <= 0)
        # 1 = successional
        # 2 = multiple mutations
        # 3 = diffusion 
        # 4 = regime undetermined
        self.evoRegime_a_i = np.zeros(self.di.shape) 
        self.evoRegime_c_i = np.zeros(self.di.shape) 
        
        return None

    #------------------------------------------------------------------------------
    
    def get_stateSpaceEvoParameters(self):
        
        # calculate population parameters for each of the states in the markov chain model
        # the evolution parameters are calculated along the absolute fitness state space
        # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
        
        # loop through state space to calculate following: 
        # mutation rates, equilb. popsize, selection coefficients
        #
        # NOTE: pFix value not calculate here, but in sepearate function to that method 
        # of getting pFix values can be selected without mucking up the code here.
        #
        # NOTE: state space always order from least fit to most fit in arrays
        for ii in range(self.get_stateSpaceSize()):
            if self.absFitType == 'dEvo':
                self.state_i[ii] = -(self.get_iExt()-ii)
            elif self.absFitType == 'bEvo':
                self.state_i[ii] = ii
            
            # mutation rates (per birth per generation - NEED TO CHECK IF CORRECT)
            self.Ua_i[ii]    = self.params['UaMax']*(1 - ii/self.get_stateSpaceSize())
            self.Uc_i[ii]    = self.params['Uc']
            
            # population sizes 
            self.eq_Ni[ii]   = self.params['T']*self.eq_yi[ii]
            
            # selection coefficients ( time scale = 1 generation)
            self.sc_i[ii]    = lmFun.get_c_SelectionCoeff(self.bi[ii],self.eq_yi[ii], \
                                                          self.params['cp'],self.di[ii])
            self.sa_i[ii]    = self.params['sa']    # di/bi defined for constant selection coeff
            
        return None 
    
    #------------------------------------------------------------------------------
        
    def get_last_di(self):
        # get_last_di() calculates next d-term after di[-1], this value is 
        # occasionally need it to calculate pfix and the rate of adaption.
        
        if self.absFitType == 'dEvo':
            # get next d-term after last di, which for RM will be dOpt
            di_last = self.params['dOpt']
            
        elif self.absFitType == 'bEvo':
            # with b-evolution, di array is constant, so just use first
            di_last = self.di[0]
            
        return di_last
    
    #------------------------------------------------------------------------------
        
    def get_last_bi(self):
        # get_last_di() calculates next d-term after di[-1], this value is 
        # occasionally need it to calculate pfix and the rate of adaption.
        
        if self.absFitType == 'dEvo':
            # with d-evolution, bi array is constant, so just use first
            bi_last = self.bi[0]
            
        elif self.absFitType == 'bEvo':
            # We have to calculate the next b-increment to get next b since 
            # b-evolution is unbounded
            f0 = self.params['sa']*(self.di[-1]-1)            
            f1 = (1-self.eq_yi[-1]) * np.exp(-self.bi[-1]*self.eq_yi[-1])
            f2 = f1 / ( f1 - f0*(1-f1) )
            delta_b = np.log( f2 )/self.eq_yi[-1]
            
            bi_last = self.bi[-1] + delta_b
        
        return bi_last
    
    #%% ----------------------------------------------------------------------------
    #  List of conrete methods from MC class
    # ------------------------------------------------------------------------------

    """
    
    def get_stateSpaceSize(self):
        
        # get the size of the state space.
    
    # ------------------------------------------------------------------------------
    
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
        
    #------------------------------------------------------------------------------

    def get_stable_state_evo_parameters(self):
        # get_stable_state_evo_parameters() returns the list of evo parameters 
        # at the stable state of the MC evolution model. This can either be at 
        # the end points of the MC state space, or where vd=ve, or where vd=vc.
    
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
        # of the pop. 
        #
        # For d-evolution, dOpt is not part of the di array, so then  
        # then dExt is just the size of di array.
        # 
        # For b-evolution, bExt is always the first state where bi = d-1. 
        #
        if self.absFitType == 'dEvo':
            iExt = self.di.size
        elif self.absFitType == 'bEvo':
            iExt = 0
            
        return iExt
    
    #------------------------------------------------------------------------------
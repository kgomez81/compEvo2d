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
import evoLibraries.LotteryModel.LM_pFix_FSA as lmPfix

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
    # Ua_i    # absolute fitness mutation rate
    # Ur_i    # relative fitness mutation rate
    # eq_yi   # equilibrium density of fitness class i
    # eq_Ni   # equilibrium population size of fitness class i
    # sd_i    # selection coefficient of "d" trait beneficial mutation
    # sc_i    # selection coefficient of "c" trait beneficial mutation
    
    # state space pFix values
    # pFix_d_i = np.zeros(self.di.shape) # pFix of "d" trait beneficial mutation
    # pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
    
    # state space evolution rates
    # va_i    # rate of adaptation in absolute fitness trait alone
    # vr_i    # rate of adaptation in relative fitness trait alone
    # ve_i    # rate of fitness decrease due to environmental degradation
    
    #------------------------------------------------------------------------------
    # Class constructor
    #------------------------------------------------------------------------------
    
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
        super().get_stateSpaceEvoRates()           
        
    #------------------------------------------------------------------------------
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
            di = di+[di[-1]*(1+self.params['sa']*(di[-1]-1))]
        
        # FinallRemove dOpt from state space
        self.di = np.asarray(di[1:])
        
        # Now that the state space is defined, adjust the size of all other arrays
        # that store evolution parameters and rates
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.size) # state number
        self.Ua_i    = np.zeros(self.di.size) # absolute fitness mutation rate
        self.Ur_i    = np.zeros(self.di.size) # relative fitness mutation rate
        self.eq_yi   = np.zeros(self.di.size) # equilibrium density of fitness class i
        self.eq_Ni   = np.zeros(self.di.size) # equilibrium population size of fitness class i
        self.sd_i    = np.zeros(self.di.size) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.size) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_d_i = np.zeros(self.di.size) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.size) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.va_i    = np.zeros(self.di.size) # rate of adaptation in absolute fitness trait alone
        self.vr_i    = np.zeros(self.di.size) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.size) # rate of fitness decrease due to environmental degradation
        
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
            self.state_i[ii] = -(ii+1)
            
            # mutation rates (per birth per generation - NEED TO CHECK IF CORRECT)
            self.Ua_i[ii]    = self.params['UaMax']*float(ii+1)/self.get_iExt()
            self.Ur_i[ii]    = self.params['Ur']
            
            # population sizes and densities 
            self.eq_yi[ii]   = lmFun.get_eqPopDensity(self.params['b'],self.di[ii],yi_option)
            self.eq_Ni[ii]   = self.params['T']*self.eq_yi[ii]
            
            # selection coefficients ( time scale = 1 generation)
            self.sc_i[ii]    = lmFun.get_c_SelectionCoeff(self.params['b'],self.eq_yi[ii], \
                                                          self.params['cr'],self.di[ii])
            self.sd_i[ii]    = self.params['sa']    # di defined for constant selection coeff
            
        return None 

    #------------------------------------------------------------------------------
    
    def get_stateSpacePfixValues(self):
        
        # calculate population parameters for each of the states in the markov chain model
        # the evolution parameters are calculated along the absolute fitness state space
        # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
        
        # loop through state space to calculate following: 
        # pFix values
        for ii in range(self.di.size):
            # ----- Probability of Fixation Calculations -------
            # Expand this section alter to select different options for calculating pFix
            # 1) First step analysis, (fastest but likely not as accurate across parameter space)
            # 2) Transition matrix steady state (slower but improved accuracy, requires tuning matrix size)
            # 3) Simulation (slowest but most accurate across parameter space)
            
            # ---- First step analysis method of obtaining pFix -------
            # set up parameters/arrays for pfix calculations
            kMax = 10   # use up to 10th order term of Prob Generating function to root find pFix
            
            # pFix d-trait beneficial mutation
            # NOTE: second array entry of dArry corresponds to mutation
            if (ii == 0):
                # if at first state space, then use dOpt since it is not in the di array
                dArry = np.array( [self.di[ii], self.params['dOpt'] ] )
            else:
                # if not at first state space then evolution goes from ii -> ii-1
                dArry = np.array( [self.di[ii], self.di[ii-1]       ] )
            cArry = np.array( [1, 1] )
            
            self.pFix_c_i[ii] = lmPfix.calc_pFix_FSA(self.params['b'], \
                                                         self.params['T'], \
                                                         dArry, \
                                                         cArry, \
                                                         kMax)
            # pFix c-trait beneficial mutation
            # NOTE: second array entry of cArry corresponds to mutation
            dArry = np.array( [self.di[ii], self.di[ii]         ] )
            cArry = np.array( [1          , 1+self.params['cr'] ] )  # mutation in c-trait
            self.pFix_d_i[ii] = lmPfix.calc_pFix_FSA(self.params['b'], \
                                                         self.params['T'], \
                                                         dArry, \
                                                         cArry, \
                                                         kMax)
        return None
    
    # ------------------------------------------------------------------------------
    #  List of conrete methods from MC class
    # ------------------------------------------------------------------------------
    
    " def get_stateSpaceEvoRates(self):                                                     "
    "                                                                                       "
    "     calculate evolution parameters for each of the states in the markov chain model   "
    "     the evolution parameters are calculated along the absolute fitness state space    "
    "     beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)     "
    

    " def read_pFixOutputs(self,readFile,nStates):                                          "
    "                                                                                       "
    "     read_pFixOutputs reads the output file containing estimated pfix values           "
    "     from simulations and stores them in an array so that they can be used in          "
    "     creating figures.                                                                 "
    
    # ------------------------------------------------------------------------------
    #  Specific methods for the RM MC class
    # ------------------------------------------------------------------------------
    
    def get_iExt(self):      
        # get_iExt()  returns the last state space corresponding to extinction 
        # of the pop. Since dOpt is always zero and not part of the di array, 
        # then dExt is just the size of di array.
        
        return self.di.size
    
    #------------------------------------------------------------------------------
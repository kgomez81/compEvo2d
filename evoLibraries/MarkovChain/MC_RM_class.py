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
import LotteryModel.LM_functions as lmFun
import LotteryModel.LM_pFix_FSA as lmPfix
import RateOfAdapt.ROA_functions as roaFun

# *****************************************************************************
# Markov Chain Class - Running Out of Mutations (RM)
# *****************************************************************************

class mcEvoModel_RM:
    # class used to encaptulate all of evolution parameters for an Markov Chain (MC)
    # representing a running out of mutations evolution model.
    
    def __init__(self,params):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        self.params = params            # dictionary with evo parameters
        
        # absolute fitness landscape (array of di terms)
        self.di = self.get_absoluteFitnessClasses() 
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.shape) # state number
        self.Ua_i    = np.zeros(self.di.shape) # absolute fitness mutation rate
        self.Ur_i    = np.zeros(self.di.shape) # relative fitness mutation rate
        self.eq_yi   = np.zeros(self.di.shape) # equilibrium density of fitness class i
        self.eq_Ni   = np.zeros(self.di.shape) # equilibrium population size of fitness class i
        self.sd_i    = np.zeros(self.di.shape) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.shape) # selection coefficient of "c" trait beneficial mutation
        
        self.get_stateSpaceEvoParameters()      # update parameter arrays above
        
        # state space pFix values
        self.pFix_d_i = np.zeros(self.di.shape) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
        
        self.get_stateSpacePfixValues()         # update pFix arrays (expand later to include options)    
        
        # state space evolution rates
        self.va_i    = np.zeros(self.di.shape) # rate of adaptation in absolute fitness trait alone
        self.vr_i    = np.zeros(self.di.shape) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.shape) # rate of fitness decrease due to environmental degradation
        
        self.get_stateSpaceEvoRates()           # update evolution rate arrays above
        
    #------------------------------------------------------------------------------
    
    def get_iExt(self):
        
        # get the last state space corresponding to extinction of the pop
        # Note: di does not include dOpt, so we add 1)
        iExt = (self.di.size+1)
        
        return iExt
    
    #------------------------------------------------------------------------------
    
    def get_absoluteFitnessClasses(self):
        # get_absoluteFitnessClasses() generates the sequence of death terms
        # that represent the absolute fitness space.
        
        # Recursively calculate set of absolute fitness classes 
        dMax = self.params['b']+1
        di = [self.params['dOpt']]
        
        while (di[-1] < dMax):
            # loop until at dMax or greater (max death term for viable pop)
            di = di+[di[-1]*(1+self.params['sa']*(di[-1]-1))]
        
        # Remove dOpt from state space
        return np.asarray(di[1:])

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
            self.eq_Ni[ii]   = self.params['T']*lmFun.eq_yi[ii]
            
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
    

    #------------------------------------------------------------------------------

    def get_stateSpaceEvoRates(self):
        
        # calculate evolution parameters for each of the states in the markov chain model
        # the evolution parameters are calculated along the absolute fitness state space
        # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
        for ii in range(self.di.size):
            # absolute fitness rate of adaptation ( on time scale of generations)
            self.va_i = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sa_i[ii], \
                                               self.Ua_i[ii], \
                                               self.pFixAbs_i[ii])
                
            # relative fitness rate of adaptation ( on time scale of generations)
            self.vr_i = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sr_i[ii], \
                                               self.Ur_i[ii], \
                                               self.pFixRel_i[ii])
                
            # rate of fitness decrease due to environmental change ( on time scale of generations)
            # fitness assumed to decrease by sa = absolute fitness increment.
            self.ve_i = self.params['sa'] * self.params['R'] * lmFun.get_iterationsPerGenotypeGeneration(self.di[ii])    
            
        return None
    
    #------------------------------------------------------------------------------
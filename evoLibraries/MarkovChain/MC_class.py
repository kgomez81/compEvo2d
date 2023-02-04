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

import LotteryModel.LM_functions as lmFun
import RateOfAdapt.ROA_functions as roaFun

class mcEvoModel(ABC):
    # ABSTRACT class used to prototype the MC classes for RM and DRE
    #
    # IMPORTANT: This class is not mean to be used for anything other then for 
    # consolidating the common code of RM and DRE MC classes, i.e. something like 
    # an Abstract class
    
    
    def __init__(self,params):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        self.params = params            # dictionary with evo parameters
    
        # absolute fitness landscape (array of di terms), 
        self.di = np.array([0]) 
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.shape) # state number
        self.Ua_i    = np.zeros(self.di.shape) # absolute fitness mutation rate
        self.Ur_i    = np.zeros(self.di.shape) # relative fitness mutation rate
        self.eq_yi   = np.zeros(self.di.shape) # equilibrium density of fitness class i
        self.eq_Ni   = np.zeros(self.di.shape) # equilibrium population size of fitness class i
        self.sd_i    = np.zeros(self.di.shape) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.shape) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_d_i = np.zeros(self.di.shape) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.va_i    = np.zeros(self.di.shape) # rate of adaptation in absolute fitness trait alone
        self.vr_i    = np.zeros(self.di.shape) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.shape) # rate of fitness decrease due to environmental degradation
    
    #------------------------------------------------------------------------------
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
        "Method that defines the arrays for evolution rates at each state"
        pass
        
    #------------------------------------------------------------------------------
    # Concrete methods (common to both RM and DR MC class implementations)
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
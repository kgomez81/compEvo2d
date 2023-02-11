# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description: Defintion of DRE MC class for defining Markov Chain models
of evolution that approximate the evolution in the Bertram & Masel 2019 
variable density lottery model.
"""

# *****************************************************************************
# import libraries
# *****************************************************************************

import numpy as np
import scipy.stats as st   

import evoLibraries.MarkovChain.MC_class as mc
import evoLibraries.LotteryModel.LM_functions as lmFun
import evoLibraries.LotteryModel.LM_pFix_FSA as lmPfix
import evoLibraries.RateOfAdapt.ROA_functions as roaFun

# *****************************************************************************
# Markov Chain Class - Diminishing Returns Epistasis (DRE)
# *****************************************************************************

class mcEvoModel_DRE(mc.mcEvoModel):
    # class used to encaptulate all of evolution parameters for an Markov Chain (MC)
    # representing a diminishing returns epistasis evolution model.
    
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
    
    #------------------------------------------------------------------------------
    # Class constructor
    #------------------------------------------------------------------------------
    
    def __init__(self,params):
        
        # Load basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        super().__init__(params)            # dictionary with evo parameters
        
        # Load absolute fitness landscape (array of di terms)
        self.get_absoluteFitnessClasses() 
        
        # update parameter arrays above
        self.get_stateSpaceEvoParameters()      # update parameter arrays above
        
        # update pFix arrays (expand later to include options)    
        self.get_stateSpacePfixValues()         # update pFix arrays (expand later to include options)    
        
        # update evolution rate arrays above
        self.get_stateSpaceEvoRates()           # update evolution rate arrays above
        
    #------------------------------------------------------------------------------
    # Definitions for abstract methods
    #------------------------------------------------------------------------------
    
    def get_absoluteFitnessClasses(self):
        # get_absoluteFitnessClasses() generates the sequence of death terms
        # that represent the absolute fitness space.
        # 
        # the fitness space is contructed up a selection coefficient threshold
        #
        # Note: the state space order for DRE is reversed from RM
        
        # Recursively calculate set of absolute fitness classes 
        dMax        = self.params['b']+1
        di          = [dMax]
        getNext_di  = True
        ii          = 1         
        iiStart     = 10
        yi_option   = 1         # option 1, analytic approx of eq. density near opt
        cdf_option  = 'logCDF'
        
        # define lowerbound for selection coefficients and how small the di
        # selection coeffient should be w.r.t ci selection coefficient
        minSelCoeff_d = 1/self.params['T']   # lower bound on neutral limit   
        minSelCoeff_c = 0.1*lmFun.get_c_SelectionCoeff(self.params['b'], \
                                   lmFun.get_eqPopDensity(self.params['b'],self.params['dOpt'],yi_option), \
                                   self.params['cp'], \
                                   self.params['dOpt'])
        
        while (getNext_di):

            # get next d-term using the selected CDF
            dNext = dMax*(self.params['dOpt']/dMax)**self.mcDRE_CDF(ii,iiStart,cdf_option)
            
            selCoeff_d_ii = lmFun.get_d_SelectionCoeff(di[-1],dNext) 
            
            
            if ( (selCoeff_d_ii > minSelCoeff_d) and (selCoeff_d_ii > minSelCoeff_c) ):
                # selection coefficient above threshold, so ad dNext 
                di = di + [ dNext ]
                ii = ii + 1
            else:
                # size of selection coefficient fell below threshold
                getNext_di = False
        
        # set the di array
        self.di = np.asarray(di)
        
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
            self.state_i[ii] = ii
            
            # mutation rates (per birth per generation - NEED TO CHECK IF CORRECT)
            self.Ud_i[ii]    = self.params['Ud']
            self.Uc_i[ii]    = self.params['Uc']
            
            # population sizes and densities 
            self.eq_yi[ii]   = lmFun.get_eqPopDensity(self.params['b'],self.di[ii],yi_option)
            self.eq_Ni[ii]   = self.params['T']*self.eq_yi[ii]
            
            # selection coefficients ( time scale = 1 generation)
            self.sc_i[ii]    = lmFun.get_c_SelectionCoeff(self.params['b'],self.eq_yi[ii], \
                                                          self.params['cp'],self.di[ii])
            # calculation for d-selection coefficient cannot be performed 
            if (ii < self.get_iMax()):
                # di size include 0 index so we can only go up to di.size-1
                self.sd_i[ii]   = lmFun.get_d_SelectionCoeff(self.di[ii],self.di[ii+1])
            else:
                # we don't story the next di term due to cutoff for the threshold
                # get next d-term using log series CDF (note: dMax = di[0])
                di_last = self.get_last_di()
                
                # save the selection coefficient of next mutation.
                self.sd_i[ii]   = lmFun.get_d_SelectionCoeff(self.di[ii],di_last) 

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
            if (ii == self.get_iMax()):
                # if at first state space, then use dOpt since it is not in the di array
                dArry = np.array( [self.di[ii], self.get_last_di()  ] )
            else:
                # if not at first state space then evolution goes from ii -> ii-1
                dArry = np.array( [self.di[ii], self.di[ii+1]       ] )
                
            cArry = np.array( [1, 1] )
            
            self.pFix_d_i[ii] = lmPfix.calc_pFix_FSA(self.params['b'], \
                                                         self.params['T'], \
                                                         dArry, \
                                                         cArry, \
                                                         kMax)
            # pFix c-trait beneficial mutation
            # NOTE: second array entry of cArry corresponds to mutation
            dArry = np.array( [self.di[ii], self.di[ii]         ] )
            cArry = np.array( [1          , 1+self.params['cp'] ] )  # mutation in c-trait
            self.pFix_c_i[ii] = lmPfix.calc_pFix_FSA(self.params['b'], \
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
            self.vd_i[ii] = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sd_i[ii], \
                                               self.Ud_i[ii], \
                                               self.pFix_d_i[ii])
                
            # relative fitness rate of adaptation ( on time scale of generations)
            self.vc_i[ii] = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sc_i[ii], \
                                               self.Uc_i[ii], \
                                               self.pFix_c_i[ii])
                
            # rate of fitness decrease due to environmental change ( on time scale of generations)
            # fitness assumed to decrease by sa = absolute fitness increment.
            self.ve_i[ii] = self.get_first_sd() * self.params['R'] \
                                    * lmFun.get_iterationsPerGenotypeGeneration(self.di[ii])    
            
        return None

    # ------------------------------------------------------------------------------
    #  List of conrete methods from MC class
    # ------------------------------------------------------------------------------
    
    " def read_pFixOutputs(self,readFile,nStates):                                          "
    "                                                                                       "
    "     read_pFixOutputs reads the output file containing estimated pfix values           "
    "     from simulations and stores them in an array so that they can be used in          "
    "     creating figures.                                                                 "
    
    # ------------------------------------------------------------------------------
    #  Specific methods for the DRE MC class
    # ------------------------------------------------------------------------------

    def get_iMax(self):
        
        # get the last state space closest the optimum
        # Note: di begins with dExt <-> i=0, so the last di is di.size-1
        # and never includes dOpt
        iMax = (self.di.size-1)
        
        return iMax
    
    #------------------------------------------------------------------------------
        
    def get_last_di(self):
        # get_last_di() calculates next d-term after di[-1], this value is 
        # occasionally need it to calculate pfix and the rate of adaption.
        
        # get next d-term after last di, using log series CDF
        di_last = self.di[0]*(self.params['dOpt']/self.di[0])**st.logser.cdf(self.get_iMax()+1,self.params['alpha'])
        
        return di_last
    
    #------------------------------------------------------------------------------
    
    def get_first_sd(self):
        # get_first_sd calculates the first coefficient of the DRE fitness 
        # landscape, i.e., the selection coefficent of the "d" beneficial 
        # mutation after dExt
        
        first_sd = lmFun.get_d_SelectionCoeff(self.di[0],self.di[1])
        
        return first_sd
    
    #------------------------------------------------------------------------------
    
    def mcDRE_CDF(self,jj,jjStart,cdfOption):
        # mcDRE_CDF calculates the CDF value of a two parameter CDF function
        # alpha for decay and jjStart for offset of CDF, i.e. we use:
        #
        # F*(jj;alpha,jjStart) = ( F(jj) - F(jjStart) ) / ( 1 - F(jjStart) )
        
        if (cdfOption == 'logCDF'):
            Fjj = st.logser.cdf(jj,self.params['alpha'])
        elif (cdfOption == 'geomCDF'):
            Fjj = st.geom.cdf(jj,1-self.params['alpha'])
        else:
            Fjj = st.logser.cdf(jj,self.params['alpha'])  # default to log series CDF
        
        return Fjj
    
    #------------------------------------------------------------------------------
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
    
    # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
    # self.params     = mcEvoOptions.params         # dictionary with evo parameters
    # self.absFitType = mcEvoOptions.absFitType     # absolute fitness evolution term
    # self.yi_option  = 3         # option 1, analytic approx of eq. density (calc near opt)
    #                             # option 2, analytic approx of eq. density (calc near ext)
    #                             # option 3, numerical soltuion (default)
    #
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
    
    def __init__(self,mcEvoOptions):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        super().__init__(mcEvoOptions)            # has dictionary with evo parameters
        
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
        ii              = 0      
        getNextStates   = True   # loop flag for d-evolution case
        
        # get the intial values of bi, di, eq_yi, sa_i sequences
        [bInit,dInit,yInit,saInit]  = self.get_initStateSpaceArryEntries()
        
        bi    = [bInit ]
        di    = [dInit ]
        eq_yi = [yInit ]
        sa_i  = [saInit]
        
        # -------------------------------------------------
        # IMPORTANT: 
        # - d-evo RM defines the state space in reverse order and then flips
        #   the arrays, i.e from dOpt to dExt
        # - b-evo RM defines the state space from bExt to bMax
        # -------------------------------------------------
        while (getNextStates):
            
            [bNext,dNext,yNext,saNext,ii] = self.get_nextStateSpaceArryEntries(bi[-1],di[-1],eq_yi[-1],ii)
            
            # ###################################### #
            # check conditions to continue loop
            # ###################################### #
            if self.absFitType == 'dEvo':

                # check conditions to continue in loop for d-evo
                cond1 = (di[-1]    < self.params['b']+1)                            # max di = dExt
                cond2 = (eq_yi[-1] < 1/self.params['dOpt'] - 1/self.params['T'])    # min eq_yi is 1/T
                
                getNextStates = cond1 and cond2
                
            elif self.absFitType == 'bEvo':
                    
                # check conditions to continue in loop for b-evo
                cond1 = (bi[-1]    < self.params['bMax'])                           # b-terms upper bound
                cond2 = (eq_yi[-1] < 1/self.params['d'] - 1/self.params['T'] )      # min eq_yi is 1/T
                
                getNextStates = cond1 and cond2
                
            # if conditions met, then add new terms to bi, di, eq_yi, sa_i list
            if getNextStates:
                bi    = bi    + [bNext ]
                di    = di    + [dNext ]
                eq_yi = eq_yi + [yNext ]
                sa_i  = sa_i  + [saNext]
            
        if (self.absFitType == 'dEvo'):
            # Finally reverse order and remove dOpt from state space
            di    = di[::-1][:-1]
            bi    = bi[::-1][:-1]
            eq_yi = eq_yi[::-1][:-1]
            sa_i  = sa_i[::-1][:-1]
        
        # set the di, bi, and eq_yi arrays. Note that there is no need to 
        # trim the arrays here as with running out of mutations.
        self.di     = np.asarray(di)
        self.bi     = np.asarray(bi)
        self.eq_yi  = np.asarray(eq_yi)
        self.sa_i   = np.asarray(sa_i)
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.size) # state number
        self.Ua_i    = np.zeros(self.di.size) # absolute fitness mutation rate
        self.Uc_i    = np.zeros(self.di.size) # relative fitness mutation rate
        self.eq_Ni   = np.zeros(self.di.size) # equilibrium population size of fitness class i
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
        # get_last_bi() calculates next d-term after bi[-1], this value is 
        # occasionally need it to calculate pfix and the rate of adaption.
        
        if self.absFitType == 'dEvo':
            # with d-evolution, bi array is constant, so just use first
            bi_last = self.bi[0]
            
        elif self.absFitType == 'bEvo':
            # We have to calculate the next b-increment to get next b since 
            # b-evolution is unbounded
            delta_b = self.di[-1] * self.bi[-1] * self.params['sa'] 
            
            bi_last = self.bi[-1] + delta_b
        
        return bi_last
    
    #------------------------------------------------------------------------------
    
    def get_next_bi(self,b,d,ii):
        # get_next_bi calculates the bi term to generate a b-sequence whose   
        # selection coefficients change according to RM or DRE 
        #
        # Note: The input ii is not used to calculate the sequence of
        #       b-terms, only d, b and model parameters.
        #
        # IMPORTANT: bi terms are defined from bExt to bMax
        # 
        # ----------------------------------------
        # Select RM Model (b-increment scheme)
        # ----------------------------------------
        
        # calculate the b increment to ensure an "sa" selection coefficient
        delta_b = d * b * self.params['sa'] 
        
        # calculate the next bi term 
        next_bi = b + delta_b
        
        return next_bi
    
    #------------------------------------------------------------------------------
    
    def get_next_di(self,b,d,ii):
        # get_next_bi calculates the bi term to generate a b-sequence whose   
        # selection coefficients change according to RM or DRE 
        #
        # Note: The inputs b, and ii are not used to calculate the sequence of
        #       d-terms, only d and model parameters.
        #
        # IMPORTANT: di terms are defined from dOpt to dExt
        #
        
        # ----------------------------------------
        # Select RM Model (d-increment scheme)
        # ----------------------------------------
        
        # calculate the next di term
        next_di = d*(1+self.params['sa']*(d-1))
        
        return next_di 
    
    #----------------------------------------------------------------------------
    
    def get_initStateSpaceArryEntries(self):
        # get_initStateSpaceArryEntries() generates the initial values for the
        # bi, di, eq_yi, sa_i arrays, which together determine the absolute 
        # fitness state space.
        #
        # parameters are initialized such that equilibrium density is 
        # approximately, eq_yi = 1/T
        #
        # RM and DRE both get initialized at the same points with respect to 
        # their d- and b- sequences. The remainder of the sequences differs
        # due to definitions for get_next_bi and get_next_di
        
        if (self.absFitType == 'dEvo'):
            # NOTE: for dEvo we define the states from dOpt to dExt
            # set start values of bi, di, & eq_yi arrays
            yInit  = 1/self.params['dOpt']      # max population density
            bInit  = self.params['b']           # constant b-value 
            dInit  = self.params['dOpt']        # d-optimum
            
            # To calculate the correct selection coefficient, we need to check
            # the next d-term after dInit
            saInit = lmFun.get_d_SelectionCoeff(dInit, \
                                                self.get_next_di(bInit,dInit,0))
            
        elif (self.absFitType == 'bEvo'):
            # set start values of bi, di, & eq_yi arrays
            yInit  = 1/self.params['T']
            dInit  = self.params['d']
            bInit  = (dInit-1)/(1-yInit)
            
            # To calculate the correct selection coefficient, we need to check
            # the next b-term after dInit
            saInit = lmFun.get_b_SelectionCoeff(bInit, \
                                                self.get_next_bi(bInit, dInit, 0), \
                                                yInit,dInit)
            
        return [bInit,dInit,saInit,yInit]
    
    #----------------------------------------------------------------------------
    
    def get_nextStateSpaceArryEntries(self, bCrnt, dCrnt, yCrnt, ii):
        # get_nextStateSpaceArryEntries() computes the next entries for the bi, di
        # eq_yi, and sa_i arrays
        
        if self.absFitType == 'dEvo':
            
            # ###################################### #
            # State space definition for d-Evolution
            # ###################################### #
            dNext  = self.get_next_di(bCrnt, dCrnt, ii)                         # next di-term
            bNext  = self.params['b']                                           # next bi-term
            yNext  = lmFun.get_eqPopDensity(self.params['b'],dCrnt,self.yi_option)   # next eq_yi-term
            iiNext = ii + 1                                                     # next ii term
            
            # To calculate the correct selection coefficient, we need to check
            # the next d-term after dNext
            saNext = lmFun.get_d_SelectionCoeff(dNext,self.get_next_di(bNext,dNext,iiNext))
            
        elif self.absFitType == 'bEvo':
            
            # ###################################### #
            # State space definition for b-Evolution
            # ###################################### #
            dNext  = self.params['d']                                           # next di-term
            bNext  = self.get_next_bi(bCrnt,dCrnt,ii)                           # next bi-term
            yNext  = lmFun.get_eqPopDensity(bNext,dNext,self.yi_option)         # next eq_yi term
            iiNext = ii + 1                                                     # next ii term
            
            # To calculate the correct selection coefficient, we need to check
            # the next b-term after bNext 
            saNext = lmFun.get_b_SelectionCoeff(bNext, \
                                                self.get_next_bi(bNext,dNext,iiNext),dNext)
                    
        return [bNext,dNext,yNext,saNext,iiNext]
    
    #%% ----------------------------------------------------------------------------
    #  List of conrete methods from MC class
    # ------------------------------------------------------------------------------

    """
    
    def get_stateSpaceSize(self):
        
        # get the size of the state space.
    
    # ------------------------------------------------------------------------------
    
    def get_initStateSpaceArryEntries(self):
        # get_initStateSpaceArryEntries() generates the initial values for the
        # bi, di, eq_yi, sa_i arrays, which together determine the absolute 
        # fitness state space.
        #
        # parameters are initialized such that equilibrium density is 
        # approximately, eq_yi = 1/T
        #
        # RM and DRE both get initialized at the same points with respect to 
        # their d- and b- sequences. The remainder of the sequences differs
        # due to definitions for get_next_bi and get_next_di
    
    # ------------------------------------------------------------------------------
    
    def get_nextStateSpaceArryEntries(self, bCrnt, dCrnt, yCrnt, ii):
        # get_nextStateSpaceArryEntries() computes the next entries for the bi, di
        # eq_yi, and sa_i arrays
        
    # ------------------------------------------------------------------------------
    
    def get_stateSpacePfixValues(self):
        
        # calculate population parameters for each of the states in the markov chain model
        # the evolution parameters are calculated along the absolute fitness state space
        # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
        
    # ------------------------------------------------------------------------------
    
    def get_pfixValuesWrapperFunction(self):
        
        # wrapper function added to use parallization when calucating pfix values across
        # the state space of the MC model.
        
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
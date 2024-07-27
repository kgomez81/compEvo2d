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

# *****************************************************************************
# Markov Chain Class - Diminishing Returns Epistasis (DRE)
# *****************************************************************************

class mcEvoModel_DRE(mc.mcEvoModel):
    # class used to encaptulate all of evolution parameters for an Markov Chain (MC)
    # representing a diminishing returns epistasis evolution model.
    
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
    # bi      #absolute fitness landscape (array of bi terms), 
    # di      #absolute fitness landscape (array of di terms), 
    
    # state space evolution parameters
    # state_i # state number
    # Ua_i    # absolute fitness mutation rate
    # Uc_i    # relative fitness mutation rate
    # eq_yi   # equilibrium density of fitness class i
    # eq_Ni   # equilibrium population size of fitness class i
    # sa_i    # selection coefficient of "b" or "d" trait beneficial mutation
    # sc_i    # selection coefficient of "c" trait beneficial mutation
    
    # state space pFix values
    # pFix_a_i = np.zeros(self.di.shape) # pFix of "b" or "d" trait beneficial mutation
    # pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
    
    # state space evolution rates
    # va_i    # rate of adaptation in absolute fitness trait alone
    # vc_i    # rate of adaptation in relative fitness trait alone
    # ve_i    # rate of fitness decrease due to environmental degradation
    
    #%%----------------------------------------------------------------------------
    # Class constructor
    #------------------------------------------------------------------------------
    
    def __init__(self,mcEvoOptions):
        
        # Load basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        super().__init__(mcEvoOptions)            # has dictionary with evo parameters
        
        # Load absolute fitness landscape (array of di terms)
        self.get_absoluteFitnessClasses() 
        
        # update parameter arrays above
        self.get_stateSpaceEvoParameters()      # update parameter arrays above
        
        # update pFix arrays (expand later to include options)    
        self.get_stateSpacePfixValues()         # update pFix arrays (expand later to include options)    
        
        # update evolution rate arrays above
        self.get_stateSpaceEvoRates()           # update evolution rate arrays above
        
    #%%----------------------------------------------------------------------------
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
        ii              = 0      
        getNextStates   = True   # loop flag for d-evolution case
        
        # get the intial values of bi, di, eq_yi, sa_i sequences
        [bInit,dInit,yInit,saInit]  = self.get_initStateSpaceArryEntries()
        
        bi    = [bInit ]
        di    = [dInit ]
        eq_yi = [yInit ]
        sa_i  = [saInit]
        
        # define lowerbound for selection coefficients and how small the 
        # selection coeffients should be w.r.t ci selection coefficient
        [minSelCoeff_b, minSelCoeff_d, minSelCoeff_c] = self.get_minSelCoeffValues()
        
        while (getNextStates):
            
            [bNext,dNext,yNext,saNext,ii] = self.get_nextStateSpaceArryEntries(bi[-1],di[-1],eq_yi[-1],ii)
            
            # ###################################### #
            # check conditions to continue loop
            # ###################################### #
            if self.absFitType == 'dEvo':

                # check conditions to continue in loop for d-evo
                cond1 = (saNext    > minSelCoeff_d      )      # sa_d above threshold size 
                cond2 = (ii        < self.params['iMax'])      # max state space size not exceeded
                
                getNextStates = cond1 and cond2    
                
            elif self.absFitType == 'bEvo':
                    
                # check conditions to continue in loop for b-evo
                cond1 = (bi[-1]    < self.params['bMax']                  ) # b-terms upper bound
                cond2 = (sa_i[-1]  > minSelCoeff_b                        ) # sa-terms lower bound
                cond3 = (eq_yi[-1] < 1/self.params['d']-1/self.params['T']) # density upper bound
                cond4 = (ii        < self.params['iMax']                  ) # max number of states
                
                getNextStates = cond1 and cond2 and cond3 and cond4
            
            # if conditions met, then add new terms to bi, di, eq_yi, sa_i list
            if getNextStates:
                bi    = bi    + [bNext ]
                di    = di    + [dNext ]
                eq_yi = eq_yi + [yNext ]
                sa_i  = sa_i  + [saNext]
                
        # set the di, bi, and eq_yi arrays. Note that there is no need to 
        # trim the arrays here as with running out of mutations.
        self.di     = np.asarray(di)
        self.bi     = np.asarray(bi)
        self.eq_yi  = np.asarray(eq_yi)
        self.sa_i   = np.asarray(sa_i)
        
        # state space evolution parameters
        self.state_i = np.zeros(self.di.shape) # state number
        self.Ua_i    = np.zeros(self.di.shape) # absolute fitness mutation rate
        self.Uc_i    = np.zeros(self.di.shape) # relative fitness mutation rate
        self.eq_Ni   = np.zeros(self.di.shape) # equilibrium population size of fitness class i
        self.sc_i    = np.zeros(self.di.shape) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_a_i = np.zeros(self.di.shape) # pFix of "d" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.va_i    = np.zeros(self.di.shape) # rate of adaptation in absolute fitness trait alone
        self.vc_i    = np.zeros(self.di.shape) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.shape) # rate of fitness decrease due to environmental degradation
        
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
        # mutation rates, equilb. density, equilb. popsize, selection coefficients
        #
        # NOTE: pFix value not calculate here, but in sepearate function to that method 
        # of getting pFix values can be selected without mucking up the code here.
        for ii in range(self.di.size):
            self.state_i[ii] = ii
            
            # mutation rates (per birth per generation - NEED TO CHECK IF CORRECT)
            self.Ua_i[ii]    = self.params['Ua']
            self.Uc_i[ii]    = self.params['Uc']
            
            # population sizes 
            self.eq_Ni[ii]   = self.params['T']*self.eq_yi[ii]
            
            # c - selection coefficients ( time scale = 1 generation)
            self.sc_i[ii]    = lmFun.get_c_SelectionCoeff(self.bi[ii],self.eq_yi[ii], \
                                                          self.params['cp'],self.di[ii])
             
        return None 
    
    #------------------------------------------------------------------------------
        
    def get_last_di(self):
        # get_last_di() calculates next d-term after di[-1], this value is 
        # occasionally need it to calculate pfix and the rate of adaption.
        
        if self.absFitType == 'dEvo':
            # get next d-term after last di, using log series CDF
            di_last = self.di[0]*(self.params['dOpt']/self.di[0])**self.mcDRE_CDF(self.get_iMax()+1)
            
        elif self.absFitType == 'bEvo':
            di_last = self.params['d']
        
        return di_last
    
    #------------------------------------------------------------------------------
    
    def get_last_bi(self):
        # get_last_bi() calculates next b-term after bi[-1], this value is 
        # occasionally needed to calculate pfix and the rate of adaption.
        
        if self.absFitType == 'dEvo':
            # b-terms are constant with d-evolution
            bi_last = self.params['b']
            
        elif self.absFitType == 'bEvo':
            # compute the fixed increment associated with the model
            delta_b = self.di[-1] * self.bi[-1] * self.params['sa_0']**self.bi.size
            
            bi_last = self.bi[-1] + delta_b
        
        return bi_last
    
    #------------------------------------------------------------------------------
    
    def get_next_bi(self,b,d,ii):
        # get_next_bi calculates the bi term to generate a b-sequence whose   
        # selection coefficients change according to RM or DRE when adaptation 
        # in the b-trait is selected. Otherwise, b-values are held constant.

        # ----------------------------------------
        # Select DRE Model (b-increment scheme)
        # ----------------------------------------
        
        if (self.params['DreMod'] == 1):
            # ignore geom. decay model, and just use sb0
            delta_b = d * b * self.params['sa_0'] 
        else:
            # DRE model with alpha
            delta_b = d * b * (self.params['sa_0'] * self.params['alpha']**ii)
        
        next_bi = b + delta_b
        
        return next_bi
    
    #------------------------------------------------------------------------------
    
    def get_next_di(self,b,d,ii):
        # get_next_di calculates the di term to generate a d-sequence whose   
        # selection coefficients change according to RM or DRE when adaptation 
        # in the d-trait is selected. Otherwise, d-values are held constant.
        # 
        # Note: The inputs b, d are not used to calculated the sequence of
        #       d-terms, only ii is used, along with the model parameters.

        # ----------------------------------------
        # Select DRE Model (d-increment scheme)
        # ----------------------------------------
        
        # get the maximum d-value, note we have to 
        eq_y0 = 1/self.params['T']
        b0    = self.params['b']
        dMax  = b0*(1-eq_y0)+1
        
        # calculate the next d-value
        next_di = dMax*(self.params['dOpt']/dMax)**self.mcDRE_CDF(ii+1)
        
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
            # set start values of bi, di, & eq_yi arrays
            yInit  = 1/self.params['T']
            bInit  = self.params['b']
            dInit  = bInit*(1-yInit)+1
            
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
                                                dInit)
            
        return [bInit,dInit,yInit,saInit]
    
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
    #  List of concrete methods from MC class
    # ------------------------------------------------------------------------------
    
    """
    
    def get_stateSpaceSize(self):
        # returns the size of the state space
        return self.di.size
    
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
    
    # ------------------------------------------------------------------------------
    
    def read_pFixOutputs(self,readFile,nStates):                                          
    
         read_pFixOutputs reads the output file containing estimated pfix values
         from simulations and stores them in an array so that they can be used in          
         creating figures.  
     
    #------------------------------------------------------------------------------

    def get_stable_state_evo_parameters(self):
        # get_stable_state_evo_parameters() returns the list of evo parameters 
        # at the stable state of the MC evolution model. This can either be at 
        # the end points of the MC state space, or where vd=ve, or where vd=vc.
        
    """
    
    #%% ----------------------------------------------------------------------------
    #  Specific methods for the DRE MC class
    # ------------------------------------------------------------------------------

    def get_iMax(self):
        
        # get the last state space closest the optimum
        # Note: di begins with dExt <-> i=0, so the last di is di.size-1
        # and never includes dOpt
        iMax = (self.di.size-1)
        
        return iMax
    
    #------------------------------------------------------------------------------
    
    def get_first_sd(self):
        # get_first_sd calculates the first coefficient of the DRE fitness 
        # landscape, i.e., the selection coefficent of the "d" beneficial 
        # mutation after dExt
        
        first_sd = lmFun.get_d_SelectionCoeff(self.di[0],self.di[1])
        
        return first_sd
    
    #------------------------------------------------------------------------------
    
    def mcDRE_CDF(self,jj):
        # mcDRE_CDF calculates the CDF value of a two parameter CDF function
        # alpha for decay and jjStart for offset of CDF, i.e. we use:
        #
        # F*(jj;alpha,jjStart) = ( F(jj+jjStart) - F(jjStart) ) / ( 1 - F(jjStart) )
        #
        # cdfOption: 1 = logCDF, 2 = geomCDF
        
        alpha   = self.params['alpha']
        jjStart = self.params['jStart']
        
        if (self.params['cdfOption'] == 1):
            Fjj = (st.logser.cdf(jj+jjStart,alpha)-st.logser.cdf(jjStart,alpha))/(1-st.logser.cdf(jjStart,alpha)) 
        elif (self.params['cdfOption'] == 2):
            Fjj = (st.geom.cdf(jj+jjStart,1-alpha)-st.geom.cdf(jjStart,1-alpha))/(1-st.geom.cdf(jjStart,1-alpha))
        else:
            # default to log series CDF
            Fjj = (st.logser.cdf(jj+jjStart,alpha)-st.logser.cdf(jjStart,alpha))/(1-st.logser.cdf(jjStart,alpha)) 
        
        return Fjj
    
    #------------------------------------------------------------------------------
    
    def get_minSelCoeffValues(self):
        # get_minSelCoeffValues() returns lower bounds on selection coefficient
        # for b, d, and c muations.
        
        # For b & d traits use a lower bound equal smallest neutral limit 
        minSelCoeff_d = 1/self.params['T']   
        minSelCoeff_b = 1/self.params['T']
        
        if (self.absFitType == 'dEvo'):
            # 1/10 of the max c-selection coefficient at dOpt
            minSelCoeff_c = 0.1*lmFun.get_c_SelectionCoeff(self.params['b'], \
                                       lmFun.get_eqPopDensity(self.params['b'],self.params['dOpt'], \
                                       self.yi_option), \
                                       self.params['cp'], \
                                       self.params['dOpt'])
        elif (self.absFitType == 'bEvo'):
            # calculate the maximum c-selection value at maximum population density
            # and use 1/10 of the max c-selection coefficient derived from taking the
            # limit of eq 13 in app C with b->infinity
            minSelCoeff_c = 0.01*self.params['cp']/self.params['d']
        
        return [minSelCoeff_b,minSelCoeff_d,minSelCoeff_c]
    
    #------------------------------------------------------------------------------
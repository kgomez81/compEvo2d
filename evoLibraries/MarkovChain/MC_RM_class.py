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
import evoLibraries.RateOfAdapt.ROA_functions as roaFun

import evoLibraries.MarkovChain.MC_functions as mcFun

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
        
        # Finally remove dOpt from state space
        self.di = np.asarray(di[1:])
        
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
            self.state_i[ii] = -(ii+1)
            
            # mutation rates (per birth per generation - NEED TO CHECK IF CORRECT)
            self.Ud_i[ii]    = self.params['UdMax']*float(ii+1)/self.get_iExt()
            self.Uc_i[ii]    = self.params['Uc']
            
            # population sizes and densities 
            self.eq_yi[ii]   = lmFun.get_eqPopDensity(self.params['b'],self.di[ii],yi_option)
            self.eq_Ni[ii]   = self.params['T']*self.eq_yi[ii]
            
            # selection coefficients ( time scale = 1 generation)
            self.sc_i[ii]    = lmFun.get_c_SelectionCoeff(self.params['b'],self.eq_yi[ii], \
                                                          self.params['cp'],self.di[ii])
            self.sd_i[ii]    = self.params['sd']    # di defined for constant selection coeff
            
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
        #
        # IMPORTANT: from the popgen perspective, the rates of adaptation that are calculated
        #            will not be on the same time-scale, and therefore, some need to be 
        #            rescaled accordingly to get properly compare them to one another.
        #
        #            vc - time-scale is one generation of mutant = 1/(d_i - 1)
        #            vd - time-scale is one generation of mutant = 1/(d_{i-1} - 1) 
        #
        #            to resolve the different time scales, we set everything to the time-scale
        #            of the wild type's generation time 1/(d_i -1)
        # 
        # NOTE:      di's do not include dOpt, and dExt = d[iExt] < ... < d[1] < d[0] < dOpt. 
        # 
        for ii in range(self.di.size):
            
            # check if the current di term is the first (ii=0), in which case, a beneficial
            # mutation in d moves you to dOpt.
            if (ii == 0):
                di_curr = self.di[ii]           # wild type
                di_next = self.params['dOpt']   # mutant
            else:
                di_curr = self.di[ii]           # wild type
                di_next = self.di[ii-1]         # mutant
            
            # calculate rescaling factor to change vd from time scale of mutant lineage's generation time
            # to time-scale of wild type's generation time.
            # 
            #       rescaleFactor = (1 gen mutant)/(1 gen wild type) = ( d_i - 1 )/( d_{i-1} - 1 )
            #
            rescaleFactor_vd = lmFun.get_iterationsPerGenotypeGeneration(di_next) / \
                                    lmFun.get_iterationsPerGenotypeGeneration(di_curr)
            
            # absolute fitness rate of adaptation ( on time scale of generations - mutant in d )
            self.vd_i[ii] = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sd_i[ii], \
                                               self.Ud_i[ii], \
                                               self.pFix_d_i[ii]) * rescaleFactor_vd
                
            # relative fitness rate of adaptation ( on time scale of generations - mutant in c )
            self.vc_i[ii] = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sc_i[ii], \
                                               self.Uc_i[ii], \
                                               self.pFix_c_i[ii])
                
            # rate of fitness decrease due to environmental change ( on time scale of generations)
            # fitness assumed to decrease by the parameter se (fitness decrease).
            #
            # NOTE: We use the wild types generation time scale to rescale ve here, as with other rates.
            #       
            
            self.ve_i[ii] = self.params['se'] * self.params['R'] * lmFun.get_iterationsPerGenotypeGeneration(self.di[ii])    
            
            # Lastly, get the regime ID's of each state space. These values are used to understand
            # where the analysis breaks down
            self.evoRegime_d_i[ii]  = roaFun.get_regimeID(self.eq_Ni[ii], \
                                               self.sd_i[ii], \
                                               self.Ud_i[ii], \
                                               self.pFix_d_i[ii])
                
            self.evoRegime_c_i[ii]  = roaFun.get_regimeID(self.eq_Ni[ii], \
                                               self.sc_i[ii], \
                                               self.Uc_i[ii], \
                                               self.pFix_c_i[ii])
            
        return None
    
    # ------------------------------------------------------------------------------
    
    def get_v_intersect_state_index(self,v2):      
        # get_v_intersect_state() returns the intersection state of two evo rate arrays
        # the implementation of this method varies for RM or DRE inheriting classes. RM
        # orders states from most beneficial to least (index-wise), and DRE is reversed
        # v2 should either be self.ve_i or self.vc_i
        #
        # NOTE: vd in comparison to v2 because it is v2's relationship with vd that 
        #       we use to determine if we return the extinction class dExt, or the
        #       highest fitness class in the vd_i array.
        # 
    
        # first we need to check if 
        #   1) there is an intersection vd < v2 and vd > v2 are true in state space
        #   2) no intersection and vd >= v2 in the state space (return dExt) 
        #   3) no intersection and vd <= v2 in the state space (return fittest state)
        # we exclude extinction state because, it doesnt even make sense to try and 
        # determine if there is an intersection there (i.e. use [:-1]).
        #
        # we also want to return the intersection type, i.e.
        #   1) vd crossing v2 downward       => intersection_type = -1 (stable attractor state)
        #   2) vd crossing v2 upward         => intersection_type  = 1 (unstable state)
        #   3) vd doesn't cross or equals v2 => intersection_type = 0 (no stoch.equil.)
        # 
        # we can just use minimizers of vDiff to locate intersection points because
        # these are discrete states, and the might be several v crossings, but 
        
        # get v-differences and reorder arrays from low to high absolute fitness 
        vDiff = self.vd_i[:-1] - v2[:-1]
        vDiff = vDiff[::-1]
        
        # save an index map to later conver index calculation back to index of the
        # original vd_i array.
        idx_map = [ii for ii in range(len(v2)-1)]   # subtract 1 to remove ext class
        idx_map = idx_map[::-1]
        
        if ( (min(vDiff) < 0) and (max(vDiff) > 0) ):
            # We have some intersection, with strict sign change. So find all 
            # intersections and get the one close to extinction
            [v_cross_idx,v_cross_types] = mcFun.calculate_v_intersections(vDiff)
            
            # select the appriate v-cross to return, this will be the first
            # occurance of a cross_type = -1 (idx = indices)
            attract_cross_idxs = np.nanargmin(v_cross_types)
            
            # get the first crossing in attract_cross_idxs and map to the 
            # original index in 
            intersect_state = idx_map[v_cross_idx[attract_cross_idxs[0]]]
            intersect_type  = v_cross_types[attract_cross_idxs[0]]
            
        elif (min(vDiff) >= 0):
            # vd is globally larger then v2, so return the highest fitness class
            # in the vd_i array
            intersect_state = 0
            intersect_type = 0
            
        elif (max(vDiff) <= 0):
            # vd is globally larger then v2, so return the extinction class
            intersect_state = self.get_
            intersect_type = 0
            
            
        return [intersect_state, intersect_type]
    
    #%% ----------------------------------------------------------------------------
    #  List of conrete methods from MC class
    # ------------------------------------------------------------------------------

    """
    
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
    
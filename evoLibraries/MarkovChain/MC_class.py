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

import evoLibraries.LotteryModel.LM_pFix_FSA as lmPfix
import evoLibraries.LotteryModel.LM_functions as lmFun

import evoLibraries.RateOfAdapt.ROA_functions as roaFun

class mcEvoModel(ABC):
    # ABSTRACT class used to prototype the MC classes for RM and DRE
    #
    # IMPORTANT: This class is not mean to be used for anything other then for 
    # consolidating the common code of RM and DRE MC classes, i.e. something like 
    # an Abstract class
    
    #%% ------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self,mcEvoOptions):
        
        # Basic evolution parameters for Lottery Model (Bertram & Masel 2019)
        self.params     = mcEvoOptions.params         # dictionary with evo parameters
        self.absFitType = mcEvoOptions.absFitType     # absolute fitness evolution term
    
        # absolute fitness landscape (array of di terms or bi terms). For b 
        # evolution, di is constant arry, while bi varies. For d evolution, bi
        # is constant arry, while di varies. bi and di have same size/shape
        self.di    = np.array([0])
        self.bi    = np.array([0])
        
        # state space evolution parameters
        # terms for absolute fitness states b or d use "a" in variable name
        self.state_i = np.zeros(self.di.shape) # state number
        self.Ua_i    = np.zeros(self.di.shape) # absolute fitness mutation rate
        self.Uc_i    = np.zeros(self.di.shape) # relative fitness mutation rate
        self.eq_yi   = np.zeros(self.di.shape) # equilibrium density of fitness class i
        self.eq_Ni   = np.zeros(self.di.shape) # equilibrium population size of fitness class i
        self.sa_i    = np.zeros(self.di.shape) # selection coefficient of "d" trait beneficial mutation
        self.sc_i    = np.zeros(self.di.shape) # selection coefficient of "c" trait beneficial mutation
        
        # state space pFix values
        self.pFix_a_i = np.zeros(self.di.shape) # pFix of "d/b" trait beneficial mutation
        self.pFix_c_i = np.zeros(self.di.shape) # pFix of "c" trait beneficial mutation
        
        # state space evolution rates
        self.va_i    = np.zeros(self.di.shape) # rate of adaptation in absolute fitness trait alone
        self.vc_i    = np.zeros(self.di.shape) # rate of adaptation in relative fitness trait alone
        self.ve_i    = np.zeros(self.di.shape) # rate of fitness decrease due to environmental degradation
        
        self.evoRegime_a_i = np.zeros(self.di.shape) # regim ID identifying Successional, Mult Mut, Diffusion
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
    def get_last_di(self):
        "get_last_ai() calculates next d-term or b-term after ai[-1], this    "
        "values is occasionally need it to calculate pfix and the rate of     "
        "adaption for RM and DRE models.                                      "
        pass
    
    #------------------------------------------------------------------------------
    
    @abstractmethod
    def get_last_bi(self):
        "get_last_ai() calculates next d-term or b-term after ai[-1], this    "
        "values is occasionally need it to calculate pfix and the rate of     "
        "adaption for RM and DRE models.                                      "
        pass
    
    #%%----------------------------------------------------------------------------
    
    def get_stateSpaceSize(self):
        return self.di.size
    
    #%%----------------------------------------------------------------------------
    # Concrete methods (common to both RM and DR MC class implementations)
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
            if (ii == self.get_stateSpaceSize()-1):
                bArry = np.array( [self.bi[ii], self.get_last_bi() ] )
                dArry = np.array( [self.di[ii], self.get_last_di() ] )
            else:
                # if not at first state space then evolution goes from ii -> ii-1
                bArry = np.array( [self.bi[ii], self.bi[ii+1]       ] )
                dArry = np.array( [self.di[ii], self.di[ii+1]       ] )
                
            cArry = np.array( [1, 1] )
            
            self.pFix_a_i[ii] = lmPfix.calc_pFix_FSA(self.params, \
                                                         bArry, \
                                                         dArry, \
                                                         cArry, \
                                                         kMax)
            # pFix c-trait beneficial mutation
            # NOTE: second array entry of cArry corresponds to mutation
            bArry = np.array( [self.bi[ii], self.bi[ii]         ] )
            dArry = np.array( [self.di[ii], self.di[ii]         ] )
            cArry = np.array( [1          , 1+self.params['cp'] ] )  # mutation in c-trait
            
            self.pFix_c_i[ii] = lmPfix.calc_pFix_FSA(self.params, \
                                                         bArry, \
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
        # NOTE:      di's do not include dOpt, and dExt = d[0] < d[1] < ... < d[iMax] < ... < dOpt. 
        # 
        for ii in range(self.get_stateSpaceSize()):
            
            # check if the current di term is the last (ii=iMax), in which case, a beneficial
            # mutation in d moves you what would have been di[iMax+1] if the sequence continued.
            if (ii == self.get_stateSpaceSize()-1):
                di_curr = self.di[ii]           # wild type
                di_next = self.get_last_di()    # mutant
            else:
                di_curr = self.di[ii]           # wild type
                di_next = self.di[ii+1]         # mutant
            
            # calculate rescaling factor to change vd from time scale of mutant lineage's generation time
            # to time-scale of wild type's generation time. The rescale factor changes when using dEvo
            # as the MC model type, otherwise it is constant
            # 
            #       rescaleFactor = (1 gen mutant)/(1 gen wild type) = ( d_i - 1 )/( d_{i-1} - 1 )
            #
            rescaleFactor = lmFun.get_iterationsPerGenotypeGeneration(di_next) / \
                                    lmFun.get_iterationsPerGenotypeGeneration(di_curr)
            
            # absolute fitness rate of adaptation ( on time scale of generations - mutant in d/b )
            self.va_i[ii] = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sa_i[ii], \
                                               self.Ua_i[ii], \
                                               self.pFix_a_i[ii]) * rescaleFactor
                
            # relative fitness rate of adaptation ( on time scale of generations - mutant in c )
            self.vc_i[ii] = roaFun.get_rateOfAdapt(self.eq_Ni[ii], \
                                               self.sc_i[ii], \
                                               self.Uc_i[ii], \
                                               self.pFix_c_i[ii])
                
            # rate of fitness decrease due to environmental change ( on time scale of generations)
            # fitness assumed to decrease by absolute fitness increment.
            #
            # NOTE: We use the wild types generation time scale to rescale ve here, as with other rates.
            #       
            
            self.ve_i[ii] = self.params['se'] * self.params['R'] * lmFun.get_iterationsPerGenotypeGeneration(self.di[ii])    
            
            # Lastly, get the regime ID's of each state space. These values are used to understand
            # where the analysis breaks down
            self.evoRegime_a_i[ii]  = roaFun.get_regimeID(self.eq_Ni[ii], \
                                               self.sa_i[ii], \
                                               self.Ua_i[ii], \
                                               self.pFix_a_i[ii])
                
            self.evoRegime_c_i[ii]  = roaFun.get_regimeID(self.eq_Ni[ii], \
                                               self.sc_i[ii], \
                                               self.Uc_i[ii], \
                                               self.pFix_c_i[ii])
            
        return None
    
    #------------------------------------------------------------------------------
    
    def get_va_i_perUnitTime(self):      
        # get_vd_i_perUnitTime()  returns the set of vd_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        # NOTE: vd saved in time-scale of wild type generations
        
        va_i_perUnitTime = np.asarray([ self.va_i[ii]*(self.di[ii]-1) for ii in range(self.get_stateSpaceSize())])
        
        return va_i_perUnitTime
    
    #------------------------------------------------------------------------------
    
    def get_vc_i_perUnitTime(self):      
        # get_vc_perUnitTime()  returns the set of vc_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        
        vc_i_perUnitTime = np.asarray([ self.vc_i[ii]*(self.di[ii]-1) for ii in range(self.get_stateSpaceSize())])
        
        return vc_i_perUnitTime
    
    #------------------------------------------------------------------------------
    
    def get_ve_i_perUnitTime(self):      
        # get_ve_i_perUnitTime()  returns the set of ve_i but with respect to the 
        # time scale of the model (i.e., time per iteration).
        #
        
        ve_i_perUnitTime = np.asarray([ self.ve_i[ii]*(self.di[ii]-1) for ii in range(self.get_stateSpaceSize())])
        
        return ve_i_perUnitTime
    
    #------------------------------------------------------------------------------
    
    def get_v_intersect_state_index(self,v2):      
        # get_v_intersect_state() returns the intersection state of two evo rate arrays
        # the implementation of this method varies for RM or DRE inheriting classes. RM
        # orders states from most beneficial to least (index-wise), and DRE is reversed
        # v2 should either be self.ve_i or self.vc_i
        #
        # NOTE: va in comparison to v2 because it is v2's relationship with va that 
        #       we use to determine if we return the extinction class dExt, or the
        #       highest fitness class in the va_i array.
        # 
    
        # first we need to check if 
        #   1) there is an intersection va < v2 and va > v2 are true in state space
        #   2) no intersection and va >= v2 in the state space (return dExt) 
        #   3) no intersection and va <= v2 in the state space (return fittest state)
        # we exclude extinction state because, it doesnt even make sense to try and 
        # determine if there is an intersection there (i.e. use [:-1]).
        #
        # we also want to return the intersection type, i.e.
        #   1) va crossing v2 downward       => intersection_type = -1 (stable attractor state)
        #   2) va crossing v2 upward         => intersection_type = 1 (unstable state)
        #   3) va doesn't cross or equals v2 => intersection_type = 0 (no stoch.equil.)
        # 
        # we can just use minimizers of vDiff to locate intersection points because
        # these are discrete states, and the might be several v crossings, but 
        
        # get v-differences 
        vDiff = self.va_i[1:] - v2[1:]
        
        # save an index map 
        idx_map = [*range(1,self.get_stateSpaceSize(),1)]  # start from 1 to remove ext class
        
        if (np.nanmin(vDiff) >= 0):
            # vd is globally larger then v2, so return the highest fitness class
            # in the vd_i array
            intersect_state = self.get_stateSpaceSize()-1
            intersect_type = 0
            
        elif (np.nanmax(vDiff) <= 0):
            # vd is globally larger then v2, so return the extinction class
            intersect_state = 0
            intersect_type = 0
            
        else:
            # We have some intersection, with strict sign change. So find all 
            # intersections and get the one close to extinction
            [v_cross_idx,v_cross_types] = mcFun.calculate_v_intersections(vDiff)
            
            # select the appriate v-cross to return, this will be the first
            # occurance of a cross_type = -1 (idx = indices)
            attract_cross_idxs = np.where(v_cross_types == -1)[0]
            
            # get the first crossing in attract_cross_idxs and map to the 
            # original index in 
            intersect_state = idx_map[v_cross_idx[attract_cross_idxs[0]]]
            intersect_type  = v_cross_types[attract_cross_idxs[0]]
            
        return [intersect_state, intersect_type]
        
    #------------------------------------------------------------------------------

    def get_va_ve_intersection_index(self):      
        # get_vd_ve_intersection() returns the state for which vd and ve are closest.
        # Serves as a wrapper for generic method get_v_intersect_state_index()
        
        # NOTE: we are calculating the index of the stable state, not the stable state
        [iStableState_index,iStableState_crossType] = self.get_v_intersect_state_index(self.ve_i)
        
        return [iStableState_index,iStableState_crossType]
    
    #------------------------------------------------------------------------------
    
    def get_va_vc_intersection_index(self):      
        # get_vd_ve_intersection() returns the state for which vd and vc are closest
        # Serves as a wrapper for generic method get_v_intersect_state_index()
        
        # NOTE: we are calculating the index of the stable state, not the stable state
        [iStableState_index,iStableState_crossType] = self.get_v_intersect_state_index(self.vc_i)        
        
        # print('------------------------------------')
        # print(iStableState_index)
        # print(iStableState_crossType)
        # print(self.vd_i-self.vc_i)
        
        return [iStableState_index,iStableState_crossType]
    
    # ------------------------------------------------------------------------------
    
    def get_mc_stable_state_idx(self):      
        # get_mc_stable_state() returns the MC stochastically stable absolute
        # fitness state. This will be whatever v-intersection is reached first
        # from the extinction state.
        
        # calculate intersection states (SS = Stable State)
        idx_SS = [self.get_va_ve_intersection_index()[0], self.get_va_vc_intersection_index()[0]]
        
        # find the intersection state closest to extinction, which requires taking
        # taking max of the two intersection states.
        mc_stable_state_idx = np.min( idx_SS )
        
        return mc_stable_state_idx
    
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
        #
        # NOTE: calculation of rho only makes sense at the intersection of 
        #       vd and vc, not necessarily at the stable state.
        
        [va_vc_intersect_idx,va_vc_intersect_type] = self.get_va_vc_intersection_index()
        
        # Check the intersection to see if it is valid. To summarize, algorithm above
        # provides the following (see MC_functions.calculate_v_intersections for details):
        # 
        # va_vc_intersect_idx       = index of first type -1 crossing, va crosses below v
        #                             occuring between endpoints of state space.
        #
        # va_vc_intersect_type      = -1
        
        # NOW CHECK FOR FOLLOWING CONDITIONS to assign a NaN for rho:
        #
        # 1) intersection is not at the start or end of the state space (proper crossing)
        #    this should not happen, but we check anyways.
        crssCond1 = (np.abs(va_vc_intersect_type) == 2) or (va_vc_intersect_idx == 0) \
                    or (va_vc_intersect_idx == self.get_stateSpaceSize()-1)
            
        # 2) The intersection type is at a Diffusive Mutation regime location
        crssCond2 = (self.evoRegime_a_i[va_vc_intersect_idx] == 3) or (self.evoRegime_c_i[va_vc_intersect_idx] == 3)
        
        # 3) points above and below (abs fit scale) are also in MM regime
        #    We check the band around the intersection +/- 5 steps
        ilw = va_vc_intersect_idx - min([10,va_vc_intersect_idx])
        ihi = va_vc_intersect_idx + min([10,self.get_stateSpaceSize()-1-va_vc_intersect_idx])
        
        crssCond3 = ( min(self.evoRegime_a_i[ilw:ihi]) < 2 ) or ( max(self.evoRegime_a_i[ilw:ihi]) > 2  )\
                    or ( min(self.evoRegime_c_i[ilw:ihi]) < 2 ) or ( max(self.evoRegime_c_i[ilw:ihi]) > 2  )
        
        if crssCond1 or crssCond2 or crssCond3:
            # bad intersection, don't calc rho            
            rho = np.nan
        else:
            # build evo param dictionary for absolute fitness trait
            evoParams_absTrait = {'s'     : self.sa_i[va_vc_intersect_idx], \
                                  'pFix'  : self.pFix_a_i[va_vc_intersect_idx], \
                                  'U'     : self.Ua_i[va_vc_intersect_idx], \
                                  'regID' : self.evoRegime_a_i[va_vc_intersect_idx]}
            
            # build evo param dictionary for relative fitness trait
            evoParams_relTrait = {'s'     : self.sc_i[va_vc_intersect_idx], \
                                  'pFix'  : self.pFix_c_i[va_vc_intersect_idx], \
                                  'U'     : self.Uc_i[va_vc_intersect_idx], \
                                  'regID' : self.evoRegime_c_i[va_vc_intersect_idx]}
            
            rho = mcFun.calculate_evoRates_rho(self.eq_Ni[va_vc_intersect_idx], \
                                         evoParams_absTrait, evoParams_relTrait)
                
        return rho
    
    #------------------------------------------------------------------------------

    def get_stable_state_evo_parameters(self):
        # get_stable_state_evo_parameters() returns the list of evo parameters 
        # at the stable state of the MC evolution model. This can either be at 
        # the end points of the MC state space, or where vd=ve, or where vd=vc.
        
        stable_state_idx = self.get_mc_stable_state_idx()
        
        # build evo param dictionary for absolute fitness trait
        params_intersect = {'eqState': self.state_i[stable_state_idx], \
                            'N'      : self.eq_Ni[stable_state_idx], \
                            'y'      : self.eq_yi[stable_state_idx], \
                            'b'      : self.bi[stable_state_idx], \
                            'd'      : self.di[stable_state_idx], \
                            'sa'     : self.sa_i[stable_state_idx], \
                            'sc'     : self.sc_i[stable_state_idx], \
                            'pFix_a' : self.pFix_a_i[stable_state_idx], \
                            'pFix_c' : self.pFix_c_i[stable_state_idx], \
                            'Ua'     : self.Ua_i[stable_state_idx], \
                            'Uc'     : self.Uc_i[stable_state_idx], \
                            'va'     : self.va_i[stable_state_idx], \
                            'vc'     : self.vc_i[stable_state_idx], \
                            've'     : self.ve_i[stable_state_idx], \
                            'regID_a': self.evoRegime_a_i[stable_state_idx], \
                            'regID_c': self.evoRegime_c_i[stable_state_idx]}
        
        return params_intersect
    
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
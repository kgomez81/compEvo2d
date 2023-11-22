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
import scipy.stats as st
import scipy.integrate as spInt
import scipy.optimize as opt
from numpy.polynomial import Polynomial

from evoLibraries.LotteryModel import LM_functions as lmFun

#------------------------------------------------------------------------------
#   Functions to calculate the probability of fixation (First Step Analysis)
#------------------------------------------------------------------------------

def pmf_1tok_MutantAdults(k,dm,lMut):
    # pmf_singleMutantAdult() provide the probability of k mutant adults after 
    # the competition and death phase, beginning from a single mutant adult.
    #
    # k - transition to k mutant adults from 1 mutant adult
    # dm - mutant death terms
    # lMut - rate of acq for new territories
    
    pFixLambda = (1 - 1/dm + k/lMut) * st.poisson.pmf(k,lMut/dm)
    
    return pFixLambda

#------------------------------------------------------------------------------

def calc_pFix_FSA(params,b,d,c,kMax):
    # get_pFix_dTrait_FSA() calculates the probability of fixation (pFix) using
    # the transitions probabilities associated with going from 1 mutant adult to
    # k mutant adults, and solves for pFix using first step analysis. 
    # 
    # get_pFix_FSA() assumes that the mutant lineage arises from a single 
    # beneficial mutation in the d-trait or c-trait (not both). Additionally,
    # This function assumes there are only two subpopulations, the wild type
    # and a second mutant subpopulation consisting of one adult.
    # 
    # b - array with birth term
    # T - Territory size
    # d - array with death terms
    # c - array with competition terms
    # kMax - highest order to use when solving the first step analysis polynomial
        
    # calculate the equilibrium density for the wild type pop
    # using the numerical estimate
    yi_option = 3
    compFix_option = 2
    if ( b[0] > d[0]-1 ):
        yEq = lmFun.get_eqPopDensity(b[0],d[0],yi_option)
    else:
        yEq = 0
    
    # determine the type of beneficial mutation 
    if ( (b[1] == b[0]) & (d[1] < d[0]) & (c[1] == c[0]) ):
        # benficial mutation in d-trait (simplest case)
        juvCompRateFactor = ( 1-np.exp(-b[0]*yEq) )
        
    elif ( (b[1] > b[0]) & (d[1] == d[0]) & (c[1] == c[0]) ):
        # benficial mutation in b (requires jumpfactor adjustment)
        juvCompRateFactor = (1-np.exp(-b[0]*yEq)) * b[1] / b[0]
        
    elif ( (b[1] == b[0]) & (d[1] == d[0]) & (c[1] > c[0]) ):
        # benefical mutation in c-trait
        # compute the increment for the competition term
        cr = c[1]-c[0]
        
        if (compFix_option ==  1):
            # this approximation relies on small cr << 1 (up to c~0.1). Smaller values
            # of b*yEq will improve the approx due heavier Poisson tails but this approx
            # is largely drive by cr. Larger cr requires more Z/(Z+cr) terms to be included
            # 
            # To get better approximation, compute (1+cr)*E[ Z/(Z+cr) | l_WildType ].
            # 
            # Below we include, no competitive advantage term, + 1st order linear expansion,
            # - minus 2nd order correction of expectation.
            juvCompRateFactor = ( 1-np.exp(-b[0]*yEq) ) \
                                   + cr*( 1 - (1+b[0]*yEq)*np.exp(-b[0]*yEq) ) \
                                   - cr*( (b[0]*yEq)**2*(1+cr)/(2+cr)*np.exp(-b[0]*yEq) )
                                   
        elif (compFix_option ==  2):
            # this calculation is derived from Mathematica's reduction of the infinite series
            # characterizing the expectation (see appendix of paper).
            juvCompRateFactor = b[0]*(1+c[1])(1-yEq)*spInt.quad(lambda x: np.exp(-b*(yEq*(1-x)+1/params['T'])*x**cr, 0, 1))
                                                                
    else:
        return 0
    
    if (yEq > 0):
        
        # mutant lineage's rate of acquisition for territories (one mutant adult)
        lMut_1 = ( (1-yEq)/yEq ) * juvCompRateFactor * np.exp(-b[1]/params['T'])
        
        # calculate coefficients of the first step analysis polynomial
        # include coefficients up to kMax order 
        coeffFSA = [ pmf_1tok_MutantAdults(ii,d[1],lMut_1) for ii in range(kMax+1) ]
        
        coeffFSA[1] = coeffFSA[1]-1  # subtract one for fix point equation 
        
        # define First step analysis calculate 
        polynomialFSA = Polynomial(coeffFSA) 
        
        # find root of FSA poly to get probability of extinction 
        pExt = opt.broyden1(polynomialFSA,[0.1], f_tol=1e-14)
        
        # calculate pFix 
        pFix = 1-pExt
    else:
        # Issue here is whether mutant arises before population dies off.
        # assuming no population.
        pFix = 0

    return pFix
    
#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Two trait adaptation: Relative versus absolute fitness 
@author: Kevin Gomez

Description:
Defines the basics functions used in all scripts that process matlab
data and create figures in the mutation-driven adaptation manuscript.

THIS FUNCTION STILL NEEDS TO BE COMPLETED
"""
# *****************************************************************************
# libraries
# *****************************************************************************

import numpy as np
import scipy.stats as st

from scipy.linalg import solve
from evoLibraries.LotteryModel import LM_functions as lmFun

# from sympy import Matrix
# from evoLibraries import constants as c 

#------------------------------------------------------------------------------
# transition probability functions
#------------------------------------------------------------------------------

def transProb_competitionPhase_dTerm(b,T,y,nm,k):
    # Function to calculate the transition probabilities for change in mutant
    # adult population size after competition phase.
    #
    # b - birth term
    # T - Total number of territories
    # y - equilibrium density of wild type
    # nm - size of mutant subpopulation
    # k - new size of mutant subpopulation after juvenile competition phase
    
    # calculate rate of juveniles winning new territories. This is derived in the
    # appendix and is valid for nm << U, otherwise a single territory will have 
    # several mutant juveniles (breaking the approximations in derivation).
    lambda_competitionPhase = ( (1-y)/y ) * ( 1-np.exp(-b*y) ) * nm * np.exp( -b*nm/T )
    
    # calculate the transition probability associated with going from nm mutant
    # adults to k adults. The pmf below is for the number of juveniles that win
    # new territories (equal to k-nm).
    if k-nm >= 0:
        trans_prob = st.poisson.pmf(lambda_competitionPhase,k-nm)
    else:
        trans_prob = 0
    
    return trans_prob

#------------------------------------------------------------------------------

def transProb_competitionPhase_cTerm(b,cp,T,y,nm,k,calc_option):
    # b - birth term
    # T - Total number of territories
    # y - equilibrium density of wild type
    # nm - size of mutant subpopulation
    # k - new size of mutant subpopulation after juvenile competition phase
    # calc_option - option to select how to calculate
    #               1) provides a numerical approximation of the 
    
    if k-nm >= 0:
        # First calculate key probabilities associated with mutant juvenile winning 
        # a single territory. This requires taking an expectation over pmf of the wild type
        # juvenile count.
        lw = b*y
        
        if calc_option:
            # use a numerical approximation for the expection of lambda_competition
            
            sumErr = 0.01  # pick threshold to cutoff of infinite sum in Poisson expectation
            kkMax = int( cp*(1-sumErr)/sumErr )
            
            P1 = np.asarray([ ( kk/(kk+cp) )*st.poisson.pmf(lw,kk) for kk in range(1,kkMax) ]).sum()
            P2 = 1 - st.poisson.cdf(lw,kkMax)
        
            # calculate rate of juveniles winning new territories. This is derived in the
            # appendix and is valid for nm << U, otherwise a single territory will have 
            # several mutant juveniles (breaking the approximations in derivation).
            lambda_competitionPhase =  ( (1-y)/y ) * ( (1+cp)*(P1+P2) ) * nm * np.exp( -b*nm/T )
        
        else:
            # use an analytic approximation for the expection of lambda_competition
            # Below we include, no competitive advantage term, + 1st order linear expansion,
            # - minus 2nd order correction of expectation.
            lambda_competitionPhase = ( 1-np.exp(-lw) ) \
                                       + cp*( 1 - (1+lw)*np.exp(-lw) ) \
                                       - cp*( (lw)**2*(1+cp)/(2+cp)*np.exp(-lw) )
        
        # calculate the transition probability associated with going from nm mutant
        # adults to k adults. The pmf below is for the number of juveniles that win
        # new territories (equal to k-nm).
        trans_prob = st.poisson.pmf(lambda_competitionPhase,k-nm)
    else:
        trans_prob = 0
    
    return trans_prob

#------------------------------------------------------------------------------

def transProb_deathPhase(di,nm,k):
    # nm - size of mutant subpopulation
    # k - new size of mutant subpopulation after juvenile competition phase
    # di - death term of mutant subpopulation
    
    # calculate the transition probability associated with going from nm mutant
    # adults to k adults for binomial pmf.
    trans_prob = st.binom.pmf(k,nm,1/di)
    
    return trans_prob
#------------------------------------------------------------------------------

def calc_pFix_MA(b,T,d,c,n1,n2):
    # b - birth term
    # T - Territory size
    # d - array with death terms
    # c - array with competition terms
    # n1 - max array size of transition probabilities (competition and death phase)
    # n2 - max array size of transition probabilities (solve for pFix)

    # calculate lambda rate
    yi_option = 3
    yi = lmFun.get_eqPopDensity(b,d[0],yi_option)
    
    # TRANSITION MATRIX for competition phase
    #
    # We construct the transpose of the transition matrix, and then take the 
    # transpose. Recall, MC transition matrices have columns that add 
    # to 1 when right multiplying (T*x) against distribtions x.
    Tc = np.zeros([n1,n1])
    for ii in range(n1):
        for jj in range(n1):
            if ii == 0:
                Tc[ii,jj]=np.kron(0,jj)
            else:
                # check with beneficial mutation case to use(d or c), but 
                # calculations can only be done for one type of mutation
                if ( (d[1] < d[0]) & (c[1] == c[0]) ):
                    # d trait beneficial mutation
                    Tc[ii,jj]=transProb_competitionPhase_dTerm(b,T,yi,ii,jj)
                    
                elif( (d[1] == d[0]) & (c[1] > c[0]) ):
                    # c trait beneficial mutation
                    calc_option = 0   # use the analytic approximation (1 = numerical approx)
                    cp = c[1] > c[0]
                    Tc[ii,jj]=transProb_competitionPhase_cTerm(b,cp,T,yi,ii,jj,calc_option)
                    
                else:
                    # if mutations in both traits, just exit the function.
                    return 0
    
    Tc = np.transpose(Tc)
    
    # TRANSITION MATRIX for death phase
    #
    # As above, we construct the transpose of the matrix, and then take the 
    # transpose after. Recall, MC transition matrices have columns that add 
    # to 1 when right multiplying (T*x) against distribtions x.
    Td = np.zeros([n1,n1])
    for ii in range(n1):
        for jj in range(n1):
            if ii == 0:
                Td[ii,jj]=np.kron(0,jj)
            else:
                Td[ii,jj]=transProb_deathPhase(d[1],ii,jj)
                
    Td = np.transpose(Td)
    
    # multiply the two matrices to get the full transition matrix
    Ts = np.matmul(Td,Tc) 
    
    # solve for the probability of fixation pFix by first solving for the 
    # probability of extinction pExt. 
    
    Ts = Ts[:n2, :n2]  # reduce the size of matrix product to size "n2 x n2"
    
    # assign remaining probability to state corresponding to fix
    for jj in range(Ts.shape[0]):
        colSum_jj = sum(Ts[:,jj])
        
        if (colSum_jj <= 1):
            # if already probabilities, then assign remaining weight to n2 sate
            Ts[-1,jj] = Ts[-1,jj] + colSum_jj
        else:
            # if not probabilities then normalize
            for ii in range(Ts.shape[0]):
                Ts[ii,jj] = Ts[ii,jj]/colSum_jj
    
    
    # Form the linear system to solve for pfix
    # 1. remove the first and last states from Ts
    Ts_solve = Ts[1:n2-1,1:n1-1]
    
    pfix_option = 1
    
    if pfix_option: 
        # Solve for all of the probabilities of fixing (achieve n2 state) from all 
        # states 1, 2, ... n2-1
        pjj_solve = Ts[n2,1:n2-1]
        pFix_ii = solve(Ts_solve, pjj_solve)
    
        pFix = pFix_ii[0]     # get probabability of reaching n2 from state 1.
                              # this is the estimate of pFix.
    else:
        # Solve for all of the probabilities of fixing (achieve 0 state) from all 
        # states 1, 2, ... n2-1
        pjj_solve = Ts[0,1:n2-1]
        pExt_ii = solve(Ts_solve, pjj_solve)
        
        pFix = 1-pExt_ii[0]   # get probabability of reaching n2 from state 1.
                              # this is the estimate of pFix.
                              
    return pFix

#------------------------------------------------------------------------------

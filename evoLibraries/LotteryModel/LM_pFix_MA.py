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
import LM_functions as lmFun

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

def transProb_competitionPhase_cTerm(b,cp,T,y,nm,k):
    # b - birth term
    # T - Total number of territories
    # y - equilibrium density of wild type
    # nm - size of mutant subpopulation
    # k - new size of mutant subpopulation after juvenile competition phase
    
    if k-nm >= 0:
        # First calculate key probabilities associated with mutant juvenile winning 
        # a single territory. This requires taking an expectation over pmf of the wild type
        # juvenile count.
        lw = b*y
        sumErr = 0.01  # pick threshold to cutoff of infinite sum in Poisson expectation
        kkMax = int( cp*(1-sumErr)/sumErr )
        
        P1 = np.asarray([ ( kk/(kk+cp) )*st.poisson.pmf(lw,kk) for kk in range(1,kkMax) ]).sum()
        P2 = 1 - st.poisson.cdf(lw,kkMax)
        
        # calculate rate of juveniles winning new territories. This is derived in the
        # appendix and is valid for nm << U, otherwise a single territory will have 
        # several mutant juveniles (breaking the approximations in derivation).
        lambda_competitionPhase =  ( (1-y)/y ) * ( (1+cp)*(P1+P2) ) * nm * np.exp( -b*nm/T )
        
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
    
    # transition matrix for competition phase
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
                    cp = c[1] > c[0]
                    Tc[ii,jj]=transProb_competitionPhase_cTerm(b,cp,T,yi,ii,jj)
                    
                else:
                    # if mutations in both traits, just exit the function.
                    return 0
    
    # transition matrix for death phase
    Td = np.zeros([n1,n1])
    for ii in range(n1):
        for jj in range(n1):
            if ii == 0:
                Td[ii,jj]=np.kron(0,jj)
            else:
                Td[ii,jj]=transProb_deathPhase(d[1],ii,jj)
                
    # multiply the two matrices to get the full transition matrix
    Ts = np.matmul(Td,Tc) 
    
    # solve for the probability of fixation
    # NEEDS TO BE COMPLETED
    
    pFix = 0
    return pFix

#------------------------------------------------------------------------------

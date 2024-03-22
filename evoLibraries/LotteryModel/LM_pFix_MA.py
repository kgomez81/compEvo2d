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
import scipy.integrate as spInt
from evoLibraries.LotteryModel import LM_functions as lmFun

# from sympy import Matrix
# from evoLibraries import constants as c 


#------------------------------------------------------------------------------
# supporting functions
#------------------------------------------------------------------------------

def normalize_transProbMatrix(Ts,norm_option):
    # norm option
    # False = normalize columns
    # True = normalize rows
    
    if norm_option:
        Ts = np.transpose(Ts)
        
    # assign remaining probability to state corresponding to fix
    for jj in range(Ts.shape[0]):
        colSum_jj = sum(Ts[:,jj])
        
        if (colSum_jj < 1):
            # if already probabilities, then assign remaining weight to n2 state
            Ts[-1,jj] = Ts[-1,jj] + (1-colSum_jj)
        else:
            # if not probabilities then normalize
            for kk in range(Ts.shape[0]):
                Ts[kk,jj] = Ts[kk,jj]/colSum_jj
                    
    if norm_option:
        Ts = np.transpose(Ts)
        
    return Ts

#------------------------------------------------------------------------------

def calculate_ProbJuvWins(params,b,d,c,yEq):

    # calculate pfix only if the equilibrium density is strictly positive
    if (yEq > 0):

        # determine the type of beneficial mutation 
        if ( (b[1] == b[0]) & (d[1] < d[0]) & (c[1] == c[0]) ):
            # benficial mutation in d-trait (no competitive advantage)
            ProbJuvWins = ( 1-np.exp(-b[0]*yEq) ) / ( b[0]*yEq )
            
        elif ( (b[1] > b[0]) & (d[1] == d[0]) & (c[1] == c[0]) ):
            # benficial mutation in b-trait (no competitive advantage)
            ProbJuvWins = (1-np.exp(-b[0]*yEq)) / ( b[0]*yEq )
            
        elif ( (b[1] == b[0]) & (d[1] == d[0]) & (c[1] > c[0]) ):
            # benefical mutation in c-trait
            # compute the increment for the competition term
            cr = c[1]-c[0]
            
            # This expresssion uses the integral version of (1+cr)*E[ Z/(Z+cr) | l_WildType ]
            # which is derived in the appendix of paper.
            ProbJuvWins = (1+cr)*((spInt.quad(lambda x: np.exp(-b[0]*yEq*(1-x))*x**cr, 0, 1,epsabs=1e-12,epsrel=1e-12))[0])
                                   
        else:
            return 0
        
    else:
        # Issue here is whether mutant arises before population dies off.
        # assuming no population.
        return 0
    
    return ProbJuvWins

#------------------------------------------------------------------------------
# transition probability functions
#------------------------------------------------------------------------------

def transProb_competitionPhase(lMut,nm,k):
    # Function to calculate the transition probabilities for change in mutant
    # adult population size after competition phase.
    #
    # params - dictionary with evo model parameters
    # b   - array with birth term
    # d   - array with death terms
    # c   - array with competition terms
    # yEq - equilibrium density
    # k   - mutant offspring
    # nm  - mutant adults

    # calculate the transition probability associated with going from nm mutant
    # adults to k adults. The pmf below is for the number of juveniles that win
    # new territories (equal to k-nm).
    if k-nm >= 0:
        trans_prob = st.poisson.pmf(k-nm,lMut)
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

def calc_pFix_MA(params,b,d,c,n1,n2,pfix_option):
    # b - birth term
    # d - array with death terms
    # c - array with competition terms
    # n1 - max array size of transition probabilities (competition and death phase)
    # n2 - max array size of transition probabilities (solve for pFix)
    #      NOTE: n2 <= n1 is required

    # calculate lambda rate
    yi_option = 3
    yEq = lmFun.get_eqPopDensity(b[0],d[0],yi_option)
    
    # calculate mutant lineage's rate of acquisition for territories (one mutant adult)
    # note: first term gives U * P(Z_x = 1) 
    ProbJuvWins = calculate_ProbJuvWins(params,b,d,c,yEq)
    
    # TRANSITION MATRIX for competition phase
    #
    # We construct the transpose of the transition matrix, and then take the 
    # transpose. Recall, MC transition matrices have columns that add 
    # to 1 when right multiplying (T*x) against distribtions x.
    Tc = np.zeros([n1,n1])
    for ii in range(n1):
        
        # calculate mutant lineage's rate of acquisition for territories (one mutant adult)
        lMut = (1-yEq) * b[1] * ii * np.exp(-b[1]*ii/params['T']) * ProbJuvWins
        
        for jj in range(n1):
            if ii == 0:
                Tc[jj,ii]=float(ii==jj)
            else:
                # here we calculate transition ii -> jj 
                # rows will add up to 1 prior to taking the transpose
                Tc[jj,ii]=transProb_competitionPhase(lMut,ii,jj)
    
    # mathematically, we use left multiplication for transition matrices 
    # applied to distribition p, i.e. p*T = p is solved for steady state. 
    # Instead we will solve T'*p' = p'
    Tc = normalize_transProbMatrix(Tc,0)    # enforce column sum = 1
    
    # TRANSITION MATRIX for death phase
    #
    # As above, we construct the transpose of the matrix, and then take the 
    # transpose after. Recall, MC transition matrices have columns that add 
    # to 1 when right multiplying (T*x) against distribtions x.
    Td = np.zeros([n1,n1])
    for ii in range(n1):
        for jj in range(n1):
            if ii == 0:
                Td[jj,ii]=float(ii==jj)
            else:
                # here we calculate transition ii -> jj 
                Td[jj,ii]=transProb_deathPhase(d[1],ii,jj)
                
    Td = normalize_transProbMatrix(Td,0)    # enforce column sum = 1
    
    # multiply the two matrices to get the full transition matrix
    # by transposing the system of equations, 
    #    p*T1*T2 = p    equivalent to   T2'*T1'*p' = p'
    Ts = np.matmul(Tc,Td) 
    
    # solve for the probability of fixation pFix by first solving for the 
    # probability of extinction pExt. 
    
    Ts = Ts[:n2, :n2]  # reduce the size of matrix product to size "n2 x n2"
    Ts = normalize_transProbMatrix(Ts,0)
    
    # Form the linear system to solve for pfix
    # 1. remove the first and last states from Ts
    Ts_solve = np.transpose(np.eye(n2-2)-Ts[1:n2-1,1:n2-1])
    
    try:
        if pfix_option:
            # Solve for all of the probabilities of fixing (achieve n2 state) from all 
            # states 1, 2, ... n2-1
            pjj_solve = np.transpose(Ts[-1,1:n2-1])
            pFix_ii = solve(Ts_solve, pjj_solve)
            
            pFix = pFix_ii[0]     # get probabability of reaching n2 from state 1.
                                  # this is the estimate of pFix.
            
        else:
            # Solve for all of the probabilities of fixing (achieve 0 state) from all 
            # states 1, 2, ... n2-1
            pjj_solve = np.transpose(Ts[0,1:n2-1])
            pExt_ii = solve(Ts_solve, pjj_solve)
            
            pFix = 1-pExt_ii[0]   # get probabability of reaching n2 from state 1.
                                  # this is the estimate of pFix.
                                  
    except:
          pFix = 0
                                  
        
    return pFix

#------------------------------------------------------------------------------
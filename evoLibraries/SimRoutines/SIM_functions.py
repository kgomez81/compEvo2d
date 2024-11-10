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
import scipy.special
import scipy.stats as st

import bisect
import math as math 

from joblib import Parallel, delayed

#------------------------------------------------------------------------------
#                       Simulation functions
# 
#------------------------------------------------------------------------------
    
def deltnplussim(m,c,U,pop,params,d): 
    # This function calculates the number of territories are won by each of the
    # existing classes in the population. This function is written specifically
    # to calculate changes in abundances for two classes, and is specialized to 
    # help estimate pFix.
    #
    # NOTE: This function can only be used to calulate the wins for a the mutant
    #       class
    #
    # Inputs:
    # m - array with number of new propogules (unoccupied territories) per class
    # c - array with set of relative fitness class values
    # d - array with set of absolute fitness class values
    # U - Number of unoccupied territories
    #
    # Outputs:    
    # 
    
    # get array with the set of juvenile densities. 
    # 1st index is the wild type and the 2nd index is the mutant class
    l = [m[ii]/float(U) for ii in range(len(m))]
    
    # Calculate the probability of at least one mutant propagules in 
    # a territory. sf = 1-cdf is the survival function
    prob_neq_0 = st.poisson.sf(0, mu=l[1])
    
    # sample the number of territories that will have mutants propagules
    # this helps avoid drawing the full U number of territories to test 
    if U == 0:
        return np.zeros(len(m))
    
    comp_U = np.random.binomial(U,prob_neq_0)
    
    # create a range of integer values between 1 and a max value.
    # The max value is 4*l_mut. The +1 is needed to make sure that 
    # array produced include the 4*l_mut value.
    rng = np.arange(1,int(4*math.ceil(l[1]))+1)
    
    # calcualte the poisson probability density values, conditional on 
    # not drawing 0 mutants. This leads to the bottom factor 
    #
    #   exp(-l)/(1-exp(-l)) = 1/(exp(l)-1)
    #
    zt_poiss_prbs = (l[1]**rng)/((scipy.special.factorial(rng)*(np.exp(l[1]) - 1)))
    
    # draw a sample of mutants for comp_U territories
    # rng = 1, 2, 3,...,4*l[1] gives number of mutants
    # p = zt_poiss_prbs/sum gives the set of probabilityies for those choices
    comp_mut = np.random.choice(rng,p=zt_poiss_prbs/sum(zt_poiss_prbs),size=[comp_U,1])
    
    # draw a sample of wild type for comp_U territories, theser are just poisson
    # samples for each of the comp_U territories.
    comp_wld = np.random.poisson(lam=l[0],size=[comp_U,1])
    
    # stack the draws of juveniles (mut & wild type) side by side
    scatter = np.hstack([comp_wld,comp_mut])
    
    # Set the number of competitions that will need to be run to check
    # the count of won territories for each class (mut & wild type)
    Up = len(scatter)    
    
    # create an array to store wins for each class, with just two classes, this will be an 
    # array of two entries [WT, MUT]. 
    wins = np.zeros(len(m))    
    
    # create an array to store comp
    comp=np.zeros(Up);
    
    # loop through each territory
    for i in range(int(Up)):
        
        # total number competing per territory. scatter[i] returns the array ex. [8 2]
        # i.e. 8 wild type juveniles, 2 mutant juveniles
        comp[i]=sum(scatter[i]) 
        
        if comp[i]>0:            
            # Sum mi ci / Sum mi, cbar n *c, n*c + m*c, ..., Sum mi ci is lotterycmf[-1]
            # here * is the element by element product of arrays
            lotterycmf = np.cumsum(np.array(scatter[i])*c) 
            
            #random.rand random between 0-1, [0 , c1 m1, c1 m1 + c2 m2] winner based on uniform
            victor=bisect.bisect(lotterycmf,np.random.rand()*lotterycmf[-1]) 
            
            # save the winner of the territory "victor" = 0 if WT, 1 if MUT, but need to add
            # 1 to record the win. For example if currently wins = [2,3], and victor is MUT = 1
            # then wins[MUT] = 3 and below increments it to 4.
            wins[victor] = wins[victor] + 1
    return wins

#------------------------------------------------------------------------------
    
def calculate_Ri_term(m,c,U):
    # This function calculates the value of the Ri competitive term. 
    # Inputs:
    # m - array of new propogule abundances
    # c - array of relative fitness values
    # U - Number of unoccupied territories
    # Outputs:    
    # out - value of Ri term
    
    # get array with the set of juvenile densities. 
    # 1st index is the wild type and the 2nd index is the mutant class
    l = [m[ii]/float(U) for ii in range(len(m))]
    
    # get total juvenile density 
    L = sum(l)
    
    # calculate the average value of the competitive trait
    # NOTE: the m[ii] is proportional to pop[ii] so okay to use
    cbar = sum([m[ii]*c[ii] for ii in range(len(m))])/sum(m)
    
    # generate array to store the Ri term the 
    out = l
    
    # loop through the classes and calcualte the respective Ri term
    for i in range(len(l)):
        # calculate value via formula from Bertram & Masel Lottery Model paper
        out[i] = cbar*np.exp(-l[i])*(1-np.exp(-(L-l[i]))) \
                /( c[i] + ( (cbar*L-c[i]*l[i]) / (L-l[i]) ) * ( (L-1+np.exp(-L)) / (1-(1+L)*np.exp(-L) ) ) )

    return out

#------------------------------------------------------------------------------
    
def calculate_Ai_term(m,c,U):
    # This function calculates the value of the Ri competitive term
    # Inputs:
    # m - array of new propogule abundances
    # c - array of relative fitness values
    # U - Number of unoccupied territories
    # Outputs:    
    # out - value of Ai term
    
    # get array with the set of juvenile densities. 
    # 1st index is the wild type and the 2nd index is the mutant class
    l = [m[ii]/float(U) for ii in range(len(m))]
    
    # get total juvenile density 
    L = sum(l)
    
    # calculate the average value of the competitive trait
    cbar = sum([m[ii]*c[ii] for ii in range(len(m))])/sum(m)
    
    # generate array to store the Ri term the 
    out = l
    
    # loop through the classes and calcualte the respective Ri term
    for i in range(len(l)):   
        if ((1-(1+l[i])*np.exp(-l[i])) > 0):
            # use normal calculation when l[i] sufficiently large
            t1 = l[i]*(1-np.exp(-l[i]))/(1-(1+l[i])*np.exp(-l[i]))
        else:             
            # use this calculation when l[i] << 1, i.e. for a small mutant class that leads to divide by zero
            t1 = 1
                        
        out[i] = cbar*( 1-np.exp(-l[i]) ) \
                    / ( c[i]*t1 + ( (cbar*L-c[i]*l[i]) / (L-l[i]) ) * \
                      ( L*(1-np.exp(-L)) / (1-(1+L)*np.exp(-L)) - t1 ) ) 
                        
    return out

#------------------------------------------------------------------------------
    
def deltnplus(m,c,U):
    # This function calculates the deterministic incremental adults from juveniles 
    # winning territorial competitions (set of delta_n_i).
    #
    # Inputs:
    # m - array of juveniles for each class
    # c - array of competitive trait values 
    # U - number of unoccupied territories
    #
    # Outputs:
    # delta_n = array of new adults per class (delta_n_1,...,delta_n_k)
    #
    
    if sum(m)>0 and U>0:
        # calculate juvenile density
        L = sum(m)/float(U)
        
        # calculate mean relative fitness (average c value)
        cbar = sum([m[ii]*c[ii] for ii in range(len(m))])/sum(m)
        
        # return the expected number of new adults
        factor = ( np.exp(-L) + ( calculate_Ri_term(m,c,U) + calculate_Ai_term(m,c,U) ) )
                  
        newAdults = [m[ii] * factor * c[ii] for ii in range(len(m))]
    
    else:
        # if the population has gone extinct
        newAdults = [0 for ii in range(len(m))]
    
    return newAdults
    
#------------------------------------------------------------------------------
        
def popdeath(pop,di):
    # This function adjusts abundances to capture deaths as determined by the 
    # respective death terms of each class.
    #
    # Inputs:
    # pop - array with set of abundances
    # di - array with set of death terms
    #
    # Outputs:
    # newpop - array with surviving adults of each class
    #
    
    newpop = []
    
    for ii in range(len(pop)):
        # for class i, calculate the number that survive
        if pop[ii] > 0:
            newpop = newpop + [np.random.binomial(pop[ii],1/di[ii])]
        else:
            # if there aren't any individuals, then just set to 0
            newpop = newpop + [0]
        
    newpop = np.asarray(newpop)
	
    return newpop

#------------------------------------------------------------------------------
    
def simulate_popEvoSelection(params,pop,b,d,c): 
    # This function simulates the evolution of a population given the parameters
    # and starting population provided in the inputs.
	#
    # Inputs:
    # params    - dictionary with evolution parameters
    # pop       - array abundances of subpopulation 
    # d         - array with d-terms of subpopulations
    # c         - array with c-terms of subpopulations
    #
    # Outputs:
    # newpop - estimate of probability of fixation
    #
    
    # calcualte the number of unoccupied territories
    U = max([0,int(params['T'] - sum(pop))])
    
    # calculate the number of juveniles
    m = [pop[ii] * ((b[ii] * U) / params['T']) for ii in range(len(b))]

    # create array to store new pop values
    newpop = [pop[i] for i in range(len(pop))]
    # print('pop array: (%i,%i)' % tuple(pop))
    
    
    if U > 0:
        # calcualte new adults using both the deterministic equations and stochastic
        # sampling of competitions
        deter_newAdults = deltnplus(m,c,U)
        stoch_newAdults = deltnplussim(m,c,U,pop,params,d)
        
        # calculate the total number of adults per class.
        print(pop[0])
        print(deter_newAdults[0])
        newpop[0] = int(pop[0] + deter_newAdults[0])
        newpop[1] = int(pop[1] + stoch_newAdults[1])
        
    
    # calculate the number of adults that survive
    newpop = popdeath(newpop,d) 
    
    return newpop

#------------------------------------------------------------------------------

def simulate_mutantPopEvo2Extinction(params,pop,b,d,c,fixThrshld):
    # This function simulates the evolution of a population, with  parameters
    # given, until the mutant population becomes extinct.
	#
    # Inputs:
    # params    - dictionary with evolution parameters
    # pop       - array abundances of subpopulation 
    # d         - array with d-terms of subpopulations
    # c         - array with c-terms of subpopulations
    # fixThreshold - minimum population size to consider lineage fixed.
    #
    # Outputs:
    # mutFixCheck - value indicating if mutant lineage fixed or not
    # time        - number of periods till extinction or fixation
    
    mutFixCheck = 0
    
    while ((pop[1] > 0) & (pop[1] < fixThrshld)): 
        pop = simulate_popEvoSelection(params,pop,b,d,c)
    
    mutFixCheck = int(pop[1] > 1)
        
    return mutFixCheck

#------------------------------------------------------------------------------

def estimate_popEvo_pFix(params,init_pop,b,d,c,nPfix,fixThrshld):
    # This function simulates the evolution of a population (selection only) 
    # of a popluation with a wild type dominant genotype, and a newly appearing 
    # mutant lineage. 
    #
    # NOTE: This is not written to handle multiple lineages. Also, this can be 
    #       put into a single script to parallelize the for loop.
	#
    # Inputs:
    # params    - list of evolution parameters
    # pop       - array with abundances
    # d         - set of death terms for each class
    # c         - set of competition terms for each class
    # nPfix     - number of samples to use for estimating pFix
    #
    # Outputs:
    # pFix      - estimate of probability of fixation
    #
    
    # create an array to store results 1-mut lineage fixes, 0-mut lineage dies
    # mutFixCheck = np.zeros([nPfix]);
    
    
    # run parallel loops through nPfix instances to estimate pFix
    # mutFixCheck = Parallel(n_jobs=6)(delayed(simulate_mutantPopEvo2Extinction)(params,init_pop,b,d,c,fixThrshld) for ii in range(nPfix))
    mutFixCheck = [simulate_mutantPopEvo2Extinction(params,init_pop,b,d,c,fixThrshld) for ii in range(nPfix)]
            
    
    # estimate pFix by summing the number of times the mutant lineage grew 
    # sufficiently large (fixThrshld)
    pFixEst = np.asarray(mutFixCheck).sum()/nPfix
    
    return pFixEst

#------------------------------------------------------------------------------
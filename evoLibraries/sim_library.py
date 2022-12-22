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
import scipy.optimize as opt
import scipy.stats as st
import scipy as sp
import copy as cpy
import os.path

import bisect
import csv
import math as math
import pickle 

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
    # U - Number of unoccupied territories
    #
    # Outputs:    
    # 
    
    # get array with the set of juvenile densities. 
    # 1st index is the wild type and the 2nd index is the mutant class
    l = m/float(U)
    
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
    comp_wld =np.random.poisson(lam=l[0],size=[comp_U,1])
    
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
    l = m/float(U)      
    
    # get total juvenile density 
    L = sum(l)
    
    # calculate the average value of the competitive trait
    cbar = sum(m*c)/sum(m)
    
    # generate array to store the Ri term the 
    out = l
    
    # loop through the classes and calcualte the respective Ri term
    for i in range(len(l)):
        if ((L-l[i]) == 0):
            # calculate value via formula when there is no variation in c' terms 
            # and when L ~ l[i], cbar ~ c[i].
            out[i] = np.exp(-l[i])*(1-np.exp(-(L-l[i]))) \
                    / ( 1 + ( (L-1+np.exp(-L)) / (1-(1+L)*np.exp(-L)) ) )
        else:
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
    l = m/float(U)
    
    # get total juvenile density 
    L = sum(l)
    
    # calculate the average value of the competitive trait
    cbar=sum(m*c)/sum(m)    
    
    # generate array to store the Ri term the 
    out = l
    
    # loop through the classes and calcualte the respective Ri term
    for i in range(len(l)):   
        if ((1-(1+l[i])*np.exp(-l[i])) == 0) or ((L-l[i]) == 0):
            # use this calculation when l[i] << 1, i.e. for a small mutant class that leads to divide by zero
            out[i] = cbar*(1-np.exp(-l[i])) \
                        / ( c[i] + cbar*( L*(1-np.exp(-L) ) / ( 1-(1+L)*np.exp(-L) ) - 1 ) )
        else: 
            out[i] = cbar*( 1-np.exp(-l[i]) ) \
                        / ( np.exp(-l[i])*c[i]*l[i]*(1-np.exp(-l[i])) / (1-(1+l[i])*np.exp(-l[i])) \
                        + ( (cbar*L-c[i]*l[i]) / (L-l[i]) ) * \
                          ( L*(1-np.exp(-L)) / (1-(1+L)*np.exp(-L)) - l[i]*(1-np.exp(-l[i]))/(1-(1+l[i])*np.exp(-l[i])) ) ) 
    
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
        cbar = sum(m*c)/sum(m)
        
        # return the expected number of new adults
        return m * ( np.exp(-L) + ( calculate_Ri_term(m,c,U) + calculate_Ai_term(m,c,U) ) * (c/cbar) )
    
    else:
        # if the population has gone extinct
        return np.zeros(len(m))
    
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
    
def calculate_popEvoSelection_pFixEst(params,pop,d,c): 
    # This function simulates the evolution of a population given the parameters
    # and starting population provided in the inputs.
	#
    # Inputs:
    # 
    # Outputs:
    # newpop - estimate of probability of fixation
    #
    
    # calcualte the number of unoccupied territories

    U = max([0,int(params['T'] - sum(pop))])
    
    # calculate the number of juveniles
    m = pop * ((params['b'] * U) / params['T'])

    # create array to store new pop values
    newpop = [pop[i] for i in range(len(pop))]
    # print('pop array: (%i,%i)' % tuple(pop))
    
    # creat output files
    # ---------------------------------------------------
    if os.path.isfile('mutOffspring.txt'):
        fm = open('mutOffspring.txt','a')
    else:
        fm = open('mutOffspring.txt','w')
    
    if os.path.isfile('mutAdultPreDeath.txt'):
        fd = open('mutAdultPreDeath.txt','a')
    else:
        fd = open('mutAdultPreDeath.txt','w')
        
    if os.path.isfile('unoccupiedT.txt'):
        fu = open('unoccupiedT.txt','a')
    else:
        fu = open('unoccupiedT.txt','w')
    
    if os.path.isfile('mutAdultPostDeath.txt'):
        fp = open('mutAdultPostDeath.txt','a')
    else:
        fp = open('mutAdultPostDeath.txt','w') 
        
    if os.path.isfile('wildAdultPostDeath.txt'):
        fw = open('wildAdultPostDeath.txt','a')
    else:
        fw = open('wildAdultPostDeath.txt','w') 
    # ---------------------------------------------------
    
    if U > 0:
        # calcualte new adults using both the deterministic equations and stochastic
        # sampling of competitions
        deter_newAdults = deltnplus(m,c,U)
        stoch_newAdults = deltnplussim(m,c,U,pop,params,d)
        
        # calculate the total number of adults per class.
        newpop[0] = int(pop[0] + deter_newAdults[0])
        newpop[1] = int(pop[1] + stoch_newAdults[1])

        fm.write('mut-offspring   : %i\n' % (newpop[1]-pop[1]))
        mutOffspring = newpop[1]-pop[1]
        fd.write('adults pre-death: %i\n' % (newpop[1]))
        fu.write('U = %i\n' % int(U))
    else:
        fu.write('U = %i\n' % int(U))
    
    
    # calculate the number of adults that survive
    newpop = popdeath(newpop,d) 
    # print('post-death   : (%E, %i)' % tuple(newpop))
    fp.write('post-death      : %i\n' % (newpop[1]))
    fw.write('post-death      : %i\n' % (newpop[0]))
    
    fm.close()
    fd.close()
    fu.close()
    fp.close()
    fw.close()
    
    return newpop,mutOffspring

#------------------------------------------------------------------------------

def simulation_popEvo_pFixEst(params,init_pop,d,c,nPfix,fixThrshld):
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
    mutFixCheck = np.zeros([nPfix]);

    mutSurviveTime = [];
    maxMutPops = []
    
    
    # creat output files
    # ---------------------------------------------------
    # if os.path.isfile('mutOffspring.txt'):
    #     fm = open('mutOffspring.txt','a')
    # else:
    #     fm = open('mutOffspring.txt','w')
    
    # if os.path.isfile('mutAdultPreDeath.txt'):
    #     fd = open('mutAdultPreDeath.txt','a')
    # else:
    #     fd = open('mutAdultPreDeath.txt','w')
        
    # if os.path.isfile('unoccupiedT.txt'):
    #     fu = open('unoccupiedT.txt','a')
    # else:
    #     fu = open('unoccupiedT.txt','w')
    
    # if os.path.isfile('mutAdultPostDeath.txt'):
    #     fp = open('mutAdultPostDeath.txt','a')
    # else:
    #     fp = open('mutAdultPostDeath.txt','w') 
        
    # if os.path.isfile('wildAdultPostDeath.txt'):
    #     fw = open('wildAdultPostDeath.txt','a')
    # else:
    #     fw = open('wildAdultPostDeath.txt','w') 
        
    fm = open('mutOffspring.txt','w')
    fd = open('mutAdultPreDeath.txt','w')
    fu = open('unoccupiedT.txt','w')
    fp = open('mutAdultPostDeath.txt','w') 
    fw = open('wildAdultPostDeath.txt','w') 
        
    fm.write("-----------------\n")
    fd.write("-----------------\n")
    fu.write("-----------------\n")
    fp.write("-----------------\n")
    fw.write("-----------------\n")
    
    fm.close()
    fd.close()
    fu.close()
    fp.close()
    fw.close()
        
    # fm = open('D:\Documents\compEvo2d\dataMutOffDistr.txt','w')
    # fd = open('D:\Documents\compEvo2d\dataMutAdlDistr.txt','w')
    # fm.writelines('Density: %f' % (sum(init_pop)/params['T']))
    # fm.writelines('\n')
    # fd.writelines('Density: %f' % (sum(init_pop)/params['T']))
    # fd.writelines('\n')
    
    # loop through nPfix instances to estimate pFix
    for ii in range(nPfix):
        pop = init_pop
        time = 0
        maxMutPops = maxMutPops + [init_pop[1]]
        
        # creat output files
        # ---------------------------------------------------
        if os.path.isfile('mutOffspring.txt'):
            fm = open('mutOffspring.txt','a')
        else:
            fm = open('mutOffspring.txt','w')
        
        if os.path.isfile('mutAdultPreDeath.txt'):
            fd = open('mutAdultPreDeath.txt','a')
        else:
            fd = open('mutAdultPreDeath.txt','w')
            
        if os.path.isfile('unoccupiedT.txt'):
            fu = open('unoccupiedT.txt','a')
        else:
            fu = open('unoccupiedT.txt','w')
        
        if os.path.isfile('mutAdultPostDeath.txt'):
            fp = open('mutAdultPostDeath.txt','a')
        else:
            fp = open('mutAdultPostDeath.txt','w') 
        
        if os.path.isfile('wildAdultPostDeath.txt'):
            fw = open('wildAdultPostDeath.txt','a')
        else:
            fw = open('wildAdultPostDeath.txt','w') 
            
        fm.write("-----------------\n")
        fd.write("-----------------\n")
        fu.write("-----------------\n")
        fp.write("-----------------\n")
        fw.write("-----------------\n")
        
        fm.close()
        fd.close()
        fu.close()
        fp.close()
        fw.close()
        # ---------------------------------------------------
        
        # print('start-gamma  : %f' % (np.sum(pop)/params['T']))
        while ((pop[1] > 0) & (pop[1] < fixThrshld)): 
            pop,mut = calculate_popEvoSelection_pFixEst(params,pop,d,c)
            time = time+1
            
        mutFixCheck[ii] = int(pop[1] > 1)
    
    # estimate pFix by summing the number of times the mutant lineage grew 
    # sufficiently large (fixThrshld)
    pFixEst = mutFixCheck.sum()/nPfix
    
    # fm.close()
    # fd.close()
    
    return pFixEst

#------------------------------------------------------------------------------
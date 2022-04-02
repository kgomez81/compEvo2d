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
import copy

import bisect
import csv
import math as math


# *****************************************************************************
# FUNCTIONS TO GET QUANTITIES FROM DESAI AND FISHER 2007
# *****************************************************************************

def get_vDF(N,s,U):
    # Calculates the rate of adaptation v, derived in Desai and Fisher 2007
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U)
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    #
    # Output: 
    # v - rate of adaptation
        
    v = s**2*(2*np.log(N*s)-np.log(s/U))/(np.log(s/U)**2)
    
    return v

#------------------------------------------------------------------------------

def get_qDF(N,s,U):
    # Calculates the rate of adaptation v, derived in Desai and Fisher 2007
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U)
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    #
    # Output: 
    # v - rate of adaptation
        
    q = 2*np.log(N*s)/np.log(s/U)
    
    return q

#------------------------------------------------------------------------------

def get_vDF_pFix(N,s,U,pFix):
    # Calculates the rate of adaptation v, using a heuristic version of Desai 
    # and Fisher 2007 argument, but without the asumption that p_fix ~ s, i.e. 
    # for a given population size (N), selection coefficient (s) and beneficial
    # mutation rate (U) and finally p_fix
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix -  probability of fixation
    #
    # Output: 
    # v - rate of adaptation
        
    v = s**2*(2*np.log(N*pFix)-np.log(s/U))/(np.log(s/U)**2)
    
    return v


#------------------------------------------------------------------------------

def get_vSucc_pFix(N,s,U,pFix):
    # Calculates the rate of adaptation v for the successional regime
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix - probability of fixation
    #
    # Output: 
    # v - rate of adaptation
        
    v = N*U*pFix*s
    
    return v

#------------------------------------------------------------------------------

def get_rateOfAdapt(N,s,U,pFix):
    # Calculates the rate of adaptation v, but checks which regime applies.
    #
    # Inputs:
    # N - population size
    # s - selection coefficient
    # U - beneficial mutation rate
    # pFix -  probability of fixation
    #
    # Output: 
    # v - rate of adaptation
    #
    
    # check that selection coefficient, pop size and beneficial mutation rate 
    # are valid parameters
    if (s <= 0) or (N <= 0) or (U <= 0) or (pFix <= 0):
        v = 0
    else:    
        # Calculate mean time between establishments
        Test = 1/N*U*pFix
        
        # Calculate mean time of sweep
        Tswp = (1/s)*np.log(N*pFix)
        
        # calculate rate of adaptation based on regime
        if (Test >= Tswp):
            v = get_vSucc_pFix(N,s,U,pFix)
        else:
            # this needs to be divided into multiple mutations and diffusion regime
            v = get_vDF_pFix(N,s,U,pFix)
    
    return v

#------------------------------------------------------------------------------

def get_eqPopDensity(b,di,option):
    # Calculate the equilibrium population size for the Bertram & Masel
    # variable density lottery model, single abs-fitness class case. For 
    # multiple abs-fitness classes, used dHar = harmonic mean of di weighted 
    # by frequency.
    #
    # Inputs:
    # b - juvenile birth term
    # di - death term of abs-fitness class i
    #
    # Output: 
    # eq_density - equilibrium population density
    #
    
    def eq_popsize_err(y):    
        # used in numerical approach to obtain equilibrium population density    
        return (1-y)*(1-np.exp(-b*y))-(di-1)*y

    if option == 1:
        # approximation near optimal gentotype
        eq_density = (1-np.exp(-b))/(di-np.exp(-b))+(di-1)/(di-np.exp(-b))* \
                        (np.exp(-b)-np.exp(-b*(1-np.exp(-b))/(di-np.exp(-b))))/ \
                                (di-np.exp(-b*(1-np.exp(-b))/(di-np.exp(-b))))
        
        eq_density = np.max([eq_density,0]) # ensure density >= 0
        
    elif option == 2:
        # approximation near extinction genotype
        eq_density = (b+2)/(2*b)*(1-np.sqrt(1-8*(b-di+1)/(b+2)**2))
        
        eq_density = np.max([eq_density,0]) # ensure density >= 0
        
    else:
        # numerical solution to steady state population size equation
        eq_density = opt.broyden1(eq_popsize_err,[1], f_tol=1e-14)
        
        eq_density = np.max([eq_density[0],0]) # ensure density >= 0
        
    return eq_density

#------------------------------------------------------------------------------
    
def get_absoluteFitnessClasses(b,dOpt,sa):
    # function calculates the class for which population size has a negative 
    # growth rate in the Bertram & Masel 2019 lottery model
    #
    # inputs:
    # b - juvenile birth rate
    # dOpt - death term of optimal genotype
    # sa - selection coefficient of beneficial mutations in "d" trait
    #
    # Output: 
    # dMax - largest death term after which there is negative growth in pop size
    # di - list of death terms corresponding to the discrete absolute fit classes
    
    # Theoretical derivations show that maximum size of death term is given by 
    # value below. See Appendix A in manuscript.
    dMax = b+1
    
    # Recursively calculate set of absolute fitness classes 
    di = [dOpt]
    ii = 0
    while (di[-1] < dMax):
        # loop until stop at dMax or greater reached
        di = di+[di[ii-1]*(1+sa*(di[ii-1]-1))]
    
    di = np.asarray(di)
    iExt = int(di.shape[0]-1)
    
    return [dMax,di,iExt]

#------------------------------------------------------------------------------

def get_absoluteFitnessClassesDRE(b,dOpt,alpha,iStop):
    # function calculates the class for which population size has a negative 
    # growth rate in the Bertram & Masel 2019 lottery model
    #
    # inputs:
    # b - juvenile birth rate
    # dOpt - death term of optimal genotype
    # dStop - max value of death term to generate classes for. This 
    #         prevents from trying to generate an infinite number of 
    #         classes
    # alpha - strength of diminishing returns epistasis
    #
    # Output: 
    # dMax - largest death term after which there is negative growth in pop size
    # di - list of death terms corresponding to the discrete absolute fit classes
    
    # Theoretical derivations show that maximum size of death term is given by 
    # value below. See Appendix A in manuscript.
    dMax = b+1
    
    # Recursively calculate set of absolute fitness classes 
    di = [dMax]
    ii = 1
    while (ii < iStop):
        # loop until stop at dStop or greater reached
        di = di + [ di[-1] + (dOpt-dMax)*(1-alpha)*alpha**(ii+1)]
        ii = ii + 1
    
    di = np.asarray(di)
    iMax = int(di.shape[0]-1)
    
    return [dMax,di,iMax]
    
#------------------------------------------------------------------------------
    
def get_c_SelectionCoeff(b,y,cr,d_i):
    # Calculate the "c" selection coefficient for the Bertram & Masel variable 
    # density lottery model for choice of ci = (1+sr)^i
    #
    # Inputs:
    # b - juvenile birth rate
    # y - equilibrium population density
    # cr - increase to ci from a single beneficial mutation is (1+cr)
    # d_i - death term of the mutant class
    #
    # Output: 
    # sr - selection coefficient of beneficial mutation in "c" trait
    #
    
    # check that population density is a positive number, otherwise there is
    # no evolution
    if (y <= 0):
        return 0
        
    # calculate rate of increase in frequency per iteration
    r_rel = cr*(1-y)*(1-(1+b*y)*np.exp(-b*y))/(y+(1-y)*(1-np.exp(-b)))*\
                    (b*y-1+np.exp(-b*y))/(b*y*(1-np.exp(-b*y))+cr*(1-(1+b*y)*np.exp(-b*y)))
    
    # re-scale time to get rate of increase in frequency per generation
    tau_i = get_iterationsPerGenotypeGeneration(d_i)
    
    sr = r_rel*tau_i
    
    return sr

#------------------------------------------------------------------------------
    
def get_iterationsPerGenotypeGeneration(d_i):
    # Calculate the "c" selection coefficient for the Bertram & Masel variable 
    # density lottery model for choice of ci = (1+cr)^i
    #
    # Inputs:
    # d_i - death term of the mutant class
    #
    # Output: 
    # tau_i - mean number of iterations per generation (i.e. average generation time in terms of iteration)
    #
    
    tau_i = 1/(d_i-1)
    
    return tau_i

#------------------------------------------------------------------------------

def read_parameterFile(readFile):
    # This function reads the parameter values a file and creates a dictionary
    # with the parameters and their values 

    # create array to store values and define the parameter names
    paramList = ['T','b','dOpt','sa','UaMax','Uad','cr','Ur','Urd','R']
    paramValue = np.zeros([len(paramList),1])
    
    # read values from csv file
    with open(readFile,'r') as csvfile:
        parInput = csv.reader(csvfile)
        for (line,row) in enumerate(parInput):
            paramValue[line] = float(row[0])
    
    # create dictionary with values
    paramDict = dict([[paramList[i],paramValue[i][0]] for i in range(len(paramValue))])
    
    return paramDict

#------------------------------------------------------------------------------
    
def read_parameterFileDRE(readFile):
    # This function reads the parameter values a file and creates a dictionary
    # with the parameters and their values 

    # create array to store values and define the parameter names
    paramList = ['T','b','dOpt','alpha','Ua','Uad','cr','Ur','Urd','R']
    paramValue = np.zeros([len(paramList),1])
    
    # read values from csv file
    with open(readFile,'r') as csvfile:
        parInput = csv.reader(csvfile)
        for (line,row) in enumerate(parInput):
            paramValue[line] = float(row[0])
    
    # create dictionary with values
    paramDict = dict([[paramList[i],paramValue[i][0]] for i in range(len(paramValue))])
    
    return paramDict

#------------------------------------------------------------------------------
    
def read_pFixOutputs(readFile,nStates):
    # This function reads the output file containing the estimated pfix values 
    # simulations and stores them in an array so that they can be used in 
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

def get_MChainPopParameters(params,di,iExt,yi_option):
    
    # Calculate all evolution parameters.
    state_i = []    # state number
    Ua_i    = []    # absolute fitness mutation rate
    Ur_i    = []    # relative fitness mutation rate
    eq_yi   = []    # equilibrium density of fitness class i
    eq_Ni   = []    # equilibrium population size of fitness class i
    sr_i    = []    # selection coefficient of "c" trait beneficial mutation
    sa_i    = []    # selection coefficient of "c" trait beneficial mutation
    
    # calculate population parameters for each of the states in the markov chain model
    # the evolution parameters are calculated along the absolute fitness state space
    # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
    for ii in range(1,iExt+1):
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr 
        state_i = state_i + [-ii]
        Ua_i    = Ua_i + [params['UaMax']*(float(ii)/iExt)]
        Ur_i    = Ur_i + [params['Ur']]
        eq_yi   = eq_yi + [get_eqPopDensity(params['b'],di[ii],yi_option)]
        eq_Ni   = eq_Ni + [params['T']*eq_yi[-1]]
        sr_i    = sr_i + [get_c_SelectionCoeff(params['b'],eq_yi[-1],params['cr'],di[ii])]
        sa_i    = sa_i + [params['sa']]
        
    return [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i]

#------------------------------------------------------------------------------

def get_MChainPopParametersDRE(params,di,iMax,yi_option):

    # Note: the state space for DRE must be calculated in reverse, since the di
    #       are specified in reverse.
    
    # Calculate all evolution parameters.
    state_i = []    # state number
    Ua_i    = []    # absolute fitness mutation rate
    Ur_i    = []    # relative fitness mutation rate
    eq_yi   = []    # equilibrium density of fitness class i
    eq_Ni   = []    # equilibrium population size of fitness class i
    sr_i    = []    # selection coefficient of "c" trait beneficial mutation
    sa_i    = []    # selection coefficient of "c" trait beneficial mutation
    
    # calculate population parameters for each of the states in the markov chain model
    # the evolution parameters are calculated along the absolute fitness state space
    # beginning with state 1 (1 mutation behind optimal) to iMax (max abs fitness state)
    for ii in range(1,iMax+1):
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr 
        state_i = state_i + [ii]
        Ua_i    = Ua_i + [params['Ua']]
        Ur_i    = Ur_i + [params['Ur']]
        eq_yi   = eq_yi + [get_eqPopDensity(params['b'],di[ii],yi_option)]
        eq_Ni   = eq_Ni + [params['T']*eq_yi[-1]]
        sr_i    = sr_i + [get_c_SelectionCoeff(params['b'],eq_yi[-1],params['cr'],di[ii])]
        sa_i    = sa_i + [(di[ii-1] - di[ii])/(di[ii]*(di[ii]-1))]
        
    return [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i]

#------------------------------------------------------------------------------

def get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option):
    
    # Calculate all evolution parameters.
    va_i    = []    # rate of adaptation in absolute fitness trait alone
    vr_i    = []    # rate of adaptation in relative fitness trait alone
    ve_i    = []    # rate of fitness decrease due to environmental degradation
    
    # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr 
    [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = get_MChainPopParameters(params,di,iExt,yi_option)
    
    # calculate evolution parameters for each of the states in the markov chain model
    # the evolution parameters are calculated along the absolute fitness state space
    # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
    for ii in range(1,iExt+1):
        # rates of fitness change ( on time scale of generations)
        va_i = va_i + [get_rateOfAdapt(eq_Ni[ii-1],sa_i[ii-1],Ua_i[ii-1],pFixAbs_i[ii-1][0])]
        vr_i = vr_i + [get_rateOfAdapt(eq_Ni[ii-1],sr_i[ii-1],Ur_i[ii-1],pFixRel_i[ii-1][0])]
        ve_i = ve_i + [sa_i[ii-1]*params['R']/(di[ii]-1)]          
        
    return [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i]

#------------------------------------------------------------------------------

def get_MChainEvoParametersDRE(params,di,iMax,pFixAbs_i,pFixRel_i,yi_option):
    
    # Calculate all evolution parameters.
    va_i    = []    # rate of adaptation in absolute fitness trait alone
    vr_i    = []    # rate of adaptation in relative fitness trait alone
    ve_i    = []    # rate of fitness decrease due to environmental degradation
    
    # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr 
    [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = get_MChainPopParametersDRE(params,di,iMax,yi_option)
    
    # calculate evolution parameters for each of the states in the markov chain model
    # the evolution parameters are calculated along the absolute fitness state space
    # beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
    for ii in range(1,iMax+1):
        # rates of fitness change ( on time scale of generations)
        va_i = va_i + [get_rateOfAdapt(eq_Ni[ii-1],sa_i[ii-1],Ua_i[ii-1],pFixAbs_i[ii-1][0])]
        vr_i = vr_i + [get_rateOfAdapt(eq_Ni[ii-1],sr_i[ii-1],Ur_i[ii-1],pFixRel_i[ii-1][0])]
        ve_i = ve_i + [sa_i[0]*params['R']/(di[ii]-1)]          # Environmental fitness change is constant
        
    return [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i]

#------------------------------------------------------------------------------

def get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i):
    # This function assumes that the intersection occurs in the 
    # multiple mutations regime. This quantity is irrelevant when in
    # the successional regime since there is no interference between 
    # evolution in traits.
        
    # find index that minimizes |va-vr|, but exclude extinction class (:-1)
    idxMin = np.argmin(np.abs(np.asarray(va_i[0:-1])-np.asarray(vr_i[0:-1])))
    
    sa = sa_i[idxMin]
    Ua = Ua_i[idxMin]
    sr = sr_i[idxMin]
    Ur = Ur_i[idxMin]
    
    # Definition of the rho at intersection in paper
    rho = np.abs((sr/sa)*(np.log(sa/Ua)/np.log(sr/Ur)))
    
    return rho

#------------------------------------------------------------------------------

def get_intersection_popDensity(va_i, vr_i, eq_yi):
    # function to calculate the intersection equilibrium density
        
    # find index that minimizes |va-vr| but exclude extinction class (:-1)
    idxMin = np.argmin(np.abs(np.asarray(va_i[0:-1])-np.asarray(vr_i[0:-1])))
    
    
    # Definition of the rho at intersection in paper
    yiInt = eq_yi[idxMin]
    
    return yiInt

#------------------------------------------------------------------------------
#                       Simulation functions
#------------------------------------------------------------------------------
    
def deltnplussim(m,c,U,pop,params): 
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
    
    print U
    print params['T']
    print sum(pop)
    print prob_neq_0
    
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
    print wins
    print compU
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
    
def calcualte_popEvoSelection_pFixEst(params,pop,d,c): 
    # This function simulates the evolution of a population given the parameters
    # and starting population provided in the inputs.
	#
    # Inputs:
    # 
    # Outputs:
    # newpop - estimate of probability of fixation
    #
    
    # calcualte the number of unoccupied territories
#    if sum(pop) > int(params['T']):
#        U = 0
#    else:
    U = min([0,int(params['T'] - sum(pop))])
    
    # calculate the number of juveniles
    m = pop * ((params['b'] * U) / params['T'])

    # create array to store new pop values
    newpop = pop
    
    if U > 0:
        # calcualte new adults using both the deterministic equations and stochastic
        # sampling of competitions
        deter_newAdults = deltnplus(m,c,U)
        stoch_newAdults = deltnplussim(m,c,U)
        
        # calculate the total number of adults per class.
        newpop[0] = int(pop[0] + deter_newAdults[0])
        newpop[1] = int(pop[1] + stoch_newAdults[1])
    
    # calculate the number of adults that survive
    newpop = popdeath(newpop,d) 
    
    return newpop

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

    # loop through nPfix instances to estimate pFix
    for ii in range(nPfix):
        pop = init_pop
        time = 0
        while ((pop[1] > 0) & (pop[1] < fixThrshld)): 
            pop = calcualte_popEvoSelection_pFixEst(params,pop,d,c)
            #print "time: %i, mutpopsize: %i" % (time,pop[1])
            time = time+1
        mutFixCheck[ii] = int(pop[1] > 1)
    
    # estimate pFix by summing the number of times the mutant lineage grew 
    # sufficiently large (fixThrshld)
    pFixEst = mutFixCheck.sum()/nPfix
    
    return pFixEst

#------------------------------------------------------------------------------
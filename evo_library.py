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
import scipy.optimize as opt
import csv

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

    return [dMax,di]

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

def read_parameterFile(readFile,paramList):
    # This function reads the parameter values a file and creates a dictionary
    # with the parameters and their values 

    # create array to store values
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

def get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option):
    
    # Calculate all evolution parameters.
    state_i = []    # state number
    Ua_i    = []    # absolute fitness mutation rate
    Ur_i    = []    # relative fitness mutation rate
    eq_yi   = []    # equilibrium density of fitness class i
    eq_Ni   = []    # equilibrium population size of fitness class i
    sr_i    = []    # selection coefficient of "c" trait beneficial mutation
    sa_i    = []    # selection coefficient of "c" trait beneficial mutation
    
    va_i    = []    # rate of adaptation in absolute fitness trait alone
    vr_i    = []    # rate of adaptation in relative fitness trait alone
    ve_i    = []    # rate of fitness decrease due to environmental degradation
    
    # calculate evolution parameter for each of the states in the markov chain model
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
        
        # rates of fitness change ( on time scale of generations)
        va_i = va_i + [get_rateOfAdapt(eq_Ni[-1],sa_i[-1],Ua_i[-1],pFixAbs_i[ii-1][0])]
        vr_i = vr_i + [get_rateOfAdapt(eq_Ni[-1],sr_i[-1],Ur_i[-1],pFixRel_i[ii-1][0])]
        ve_i = ve_i + [sa_i[-1]*params['R']/(di[ii]-1)]          
        
    return [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i]

#------------------------------------------------------------------------------
#                       Simulation functions
#------------------------------------------------------------------------------
    
def deltnplussim(m,c,U): 
    # This function 
    #
    # Inputs:
    # m - array with number of new propogules (unoccupied territories) per class
    # c - array with set of relative fitness class values
    # U - Number of unoccupied territories
    #
    # Outputs:    
    # 
    
    # get array with the total number of propogules per class
    l = m/float(U)      

    # sample the fraction of territories that draw zero
    prob_neq_0 = st.poisson.sf(0, mu=l[1])
    comp_U = np.random.binomial(U,prob_neq_0)
    
    rng = np.arange(1,int(4*math.ceil(l[1]))+1)
    zt_poiss_prbs = (l[1]**rng)/((scipy.special.factorial(rng)*(np.exp(l[1]) - 1)))
    
    comp_mut = np.random.choice(rng,p=zt_poiss_prbs/sum(zt_poiss_prbs),size=[comp_U,1])
    comp_wld =np.random.poisson(lam=l[0],size=[comp_U,1])
    
    scatter = np.hstack([comp_wld,comp_mut])
    Up = len(scatter)
    
    wins = np.zeros(len(m))
    comp=np.zeros(Up);
    for i in range(int(Up)):
        comp[i]=sum(scatter[i]) #total number competing per territory
        if comp[i]>0:            
            lotterycmf=np.cumsum(np.array(scatter[i])*c) # Sum mi ci / Sum mi, cbar n *c, n*c + m*c, ..., Sum mi ci is lotterycmf[-1]
            victor=bisect.bisect(lotterycmf,np.random.rand()*lotterycmf[-1]) #random.rand random between 0-1, [0 , c1 m1, c1 m1 + c2 m2] winner based on uniform
            wins[victor] = wins[victor] + 1
    return wins

#------------------------------------------------------------------------------
    
def calculate_Ri_term(m,c,U):
    # This function calculates the value of the Ri competitive term
    #
    # Inputs:
    # m - array of new propogule abundances
    # c - array of relative fitness values
    # U - 
    #
    # Outputs:    
    # out - value of Ri term
    
    l=m/float(U)
    L=sum(l)
    cbar=sum(m*c)/sum(m)
    out = l
    for i in range(len(l)):
        try:
            out[i]=cbar*np.exp(-l[i])*(1-np.exp(-(L-l[i])))\
                    /(c[i] + (L-1+np.exp(-L))/(1-(1+L)*np.exp(-L))*(cbar*L-c[i]*l[i])/(L-l[i]))
        except FloatingPointError:
            out[i]=np.exp(-l[i])*(1-np.exp(-(L-l[i])))\
                /(1 + (L-1+np.exp(-L))/(1-(1+L)*np.exp(-L)))
#    for i in range(len(out)):
#        if np.isnan(out)[i]: out[i]=0            
    return out

#------------------------------------------------------------------------------
    
def calculate_Ai_term(m,c,U):
    # This function 
    #
    # Inputs:
    # m - number of new propogules produced 
    # c - relative fitness 
    # U - 
    #
    # Outputs:    
    # 
    l=m/float(U)
    L=sum(l)
    cbar=sum(m*c)/sum(m)    
    out = l
    
    for i in range(len(l)):    
        try:
            out[i]=cbar*(1-np.exp(-l[i]))\
                    /((1-np.exp(-l[i]))/(1-(1+l[i])*np.exp(-l[i]))*c[i]*l[i]\
                    +(L*(1-np.exp(-L))/(1-(1+L)*np.exp(-L))-l[i]*(1-np.exp(-l[i]))/(1-(1+l[i])*np.exp(-l[i])))/(L-l[i])*(cbar*L-c[i]*l[i]))
        except FloatingPointError:
            out[i]=cbar*(1-np.exp(-l[i]))\
                    /(c[i]+(L*(1-np.exp(-L))/(1-(1+L)*np.exp(-L))-1))
#    for i in range(len(out)):
#        if np.isnan(out)[i]: out[i]=-1
    return out

#------------------------------------------------------------------------------
    
# This function is drawing divide by zero errors, most likely just numerical precision, but ask kevin
def deltnplus(m,c,U):
    if sum(m)>0 and U>0:
        L=sum(m)/float(U)
        cbar=sum(m*c)/sum(m)
        return m*(np.exp(-L)+(R(m,c,U)+A(m,c,U))*c/cbar)
    else:
        return np.zeros(len(m))
    
#------------------------------------------------------------------------------
        
def popdeath(pop,di):
    # goal here is the sum of the win vectors
	# then we just add them to the pop
    npop = np.array([np.random.binomial(pop[0],1/di), np.random.binomial(pop[1],1/di)]); #is the number that survive
	# we would then output this number as the new initial population and iterated
	# the process some arbitrary amount of times
    return npop

#------------------------------------------------------------------------------
    
def di_class(d,di,sa):
    return math.ceil(np.log(float(d) / float(di))/np.log(1+sa))

#------------------------------------------------------------------------------
    
def simpop(samp,steps,T,sr,b,di,do,sa,de,yi_option): #and all other shit here

	# here we need to take an initial population, with all associated parameters
	# and run through Jasons model i.e.
    c = np.array([1,(1+sr)]);
    pfix = 0;
    print(get_eq_pop_density(b,di,sa,yi_option))
    for i in range(int(samp)):
        eq_pop = math.ceil(T * get_eq_pop_density(b,di,sa,yi_option));
        #print(eq_pop/T)
        pop = np.array([eq_pop-1, 1]);

        while ((pop[1] > 0) & (pop[1] < 1000)): # 4/18: set to 1000 for now, no clue though, != 0 1/pfix calculated via mathematica, grown larger than wild
            U = int(T - sum(pop));
            mi = pop * ((b * U) / T); #or something to generate propugules numbers
            
            deltan = deltnplussim(mi,c,U)
            wins = np.array([0,deltan[1]]) # 4/18: Change to deterministic growth 
            
            npop = pop+wins;
            pop = np.array([(1/di)*(pop[0] + deltnplus(mi,c,U)[0]), np.random.binomial(npop[1],1/di)]); # 4/18: changed to deterministic
            #print(pop[1])
            
        pfix = pfix + (pop[1] > 1)/float(samp);
        
    return pfix
# Everythings running well, only issue is that it spits out nonsensical probabilities sometimes, (how the hell do you get 0.152 from 1 sample?)
#possibility of poisson probabilites not working right
#sum of mi and li.... poiss(mi) =dist= sum_{to U}[poiss(li)]

def trackpop(samp,steps,T,sr,b,di,do,sa,de,yi_option): #and all other shit here

	# here we need to take an initial population, with all associated parameters
	# and run through Jasons model i.e.
    c = np.array([1,(1+sr)]);
    print(get_eq_pop_density(b,di,sa,yi_option))
    tracker = np.array(np.zeros(steps))
    for i in range(int(samp)):
        eq_pop = math.ceil(T * get_eq_pop_density(b,di,sa,yi_option));
        #print(eq_pop/T)
        pop = np.array([eq_pop, 1]);
        trk = np.array(np.zeros(steps))
        trk[0] = 1
        its = 1 
        while ((pop[1] > 0) & (pop[1] < 100) & (its < steps)): # pop2 > 1000, while < 1000 or != 0 1/pfix calculated via mathematica, grown larger than wild
            U = int(T - sum(pop));
            mi = pop * ((b * U) / T); #or something to generate propugules numbers
            #print(mi)
            deltan = deltnplussim(mi,c,U)
            wins = np.array([deltan[0],deltan[1]])
            npop = pop+wins;
            pop = np.array([np.random.binomial(npop[0],1/di), np.random.binomial(npop[1],1/di)]);
            trk[its] = pop[1]
            its = its + 1;
        
            
        tracker = np.vstack((tracker,trk))
        
    return tracker

#------------------------------------------------------------------------------
    
def compwin(n,samp,T,sr,b,di,do,sa,de,yi_option): 

	# here we need to take an initial population, with all associated parameters
	# and run through Jasons model i.e.
    c = np.array([1,(1+sr)]);
    wins = np.array(np.zeros(samp))
    eq_pop = math.ceil(T * get_eq_pop_density(b,di,sa,yi_option));
        #print(eq_pop/T)
    pop = np.array([eq_pop-n, n]);
    U = int(T - sum(pop));
    mi = pop * ((b * U) / T); #or something to generate propugules numbers
            #print(mi)
    for i in range(samp):
        wins[i] = deltnplussim(mi,c,U)[1]
        
    return wins

#deterministic equations for wild, only consider competition when the mutant has a chance to win, everything else is a lost.  


# i.e. remove the poisson deterministic sampling for the one here: n -> m -> Bin(n + m, 1 - 1/di) = n (wrap)	
# when the approximations break down, assumptions regarding the poisson (when the bins work, binomial -> poisson) - go back through the paper.

#------------------------------------------------------------------------------
    
def modsimpop(d_Inc,c_Inc,samp,T,sr,b,dis,do,sa,de,yi_option): #and all other shit here

	# here we need to take an initial population, with all associated parameters
	# and run through Jasons model i.e.
    c = np.array([1,(1+sr*c_Inc)]);
    pfix = 0;
    #print(get_eq_pop_density(b,dis[0],sa,yi_option))
    for i in range(int(samp)):
        eq_pop = int(math.ceil(T * get_eq_pop_density(b,dis[0],sa,yi_option)));
        #print(eq_pop/T)
        pop = np.array([eq_pop-1, 1]);

        while ((pop[1] > 0) & (pop[1] < 1000)): # 4/18: set to 1000 for now, no clue though, != 0 1/pfix calculated via mathematica, grown larger than wild
            U = int(T - sum(pop));
            mi = pop * ((b * U) / T); #or something to generate propugules numbers
            deltan = deltnplussim(mi,c,U)

            pop = np.array([(1/dis[0])*(pop[0] + deltnplus(mi,c,U)[0]), np.random.binomial(pop[1]+int(deltan[1]),1/dis[d_Inc])]); # 4/18: changed to deterministic
            # print(pop[1])
            
        pfix = pfix + (pop[1] > 1)/float(samp);
        
    return pfix
#possibility of poisson probabilites not working right
#sum of mi and li.... poiss(mi) =dist= sum_{to U}[poiss(li)]
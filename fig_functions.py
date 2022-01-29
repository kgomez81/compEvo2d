# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 11:25:45 2019
Masel Lab
Project: Mutation-driven Adaptation
@author: Kevin Gomez

Description:
Defines the basics functions used in all scripts that process matlab
data and create figures in the mutation-driven adaptation manuscript.
"""
# libraries
import numpy as np
import scipy.optimize as opt

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

def get_vDF_pfix(N,s,U,pFix):
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

def get_rateOfAdapt_v(N,s,U,pFix):
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
    
    # Calculate mean time between establishments
    Test = N*U*pFix
    
    # Calculate mean time of sweep
    Tswp = (1/s)*np.log(N*pFix)
    
    # calculate rate of adaptation based on regime
    if (Test < Tswp):
        v = get_vSucc_pFix(N,s,U,pFix)
    else:
        # this needs to be divided into multiple mutations and diffusion regime
        v = get_vDF_pFix(N,s,U,pFix)
    
    return v

#------------------------------------------------------------------------------

def get_eq_pop_density(b,d,sa,i,option):
    # Calculate the equilibrium population size for the Bertram & Masel
    # variable density lottery model, single class case.
    #
    # Inputs:
    # b - juvenile birth rate
    # d - death rate
    # i - absolute fitness class (i beneficial mutations from optimal)
    # sa - absolute fitness selection coefficient
    #
    # Output: 
    # eq_density - equilibrium population density
    #
    # Remark: must have 1/sa*d > i >= 0 in general but for i = (1-d/(b+1))/sd
    # the population is in decline, unless it is rescued. Also, returns zero 
    # if formula gives negative density.
    
    di = get_class_death_rate(d,sa,i)
    
    def eq_popsize_err(y):    
        # used in numerical approach to obtain equilibrium population density    
        return (1-y)*(1-np.exp(-b*y))-(di-1)*y

    if option == 1:
        # approximation near optimal gentotype
        eq_density = (1-np.exp(-b))/(di-np.exp(-b))+(di-1)/(di-np.exp(-b))*(np.exp(-b)-np.exp(-b*(1-np.exp(-b))/(di-np.exp(-b))))/(di-np.exp(-b*(1-np.exp(-b))/(di-np.exp(-b))))
        
        eq_density = np.max([eq_density,0]) # formula should 
        
    elif option == 2:
        # approximation near extinction genotype
        eq_density = (b+2)/(2*b)*(1-np.sqrt(1-8*(b-di+1)/(b+2)**2))
        
        eq_density = np.max([eq_density,0])
        
    else:
        # numerical solution to steady state population size equation
        eq_density = opt.broyden1(eq_popsize_err,[1], f_tol=1e-14)
        
        eq_density = np.max([eq_density,0])
        
    return eq_density

#------------------------------------------------------------------------------
    
def get_extinction_classes(b,dOpt,sa):
    # function calculates the class for which population size has a negative 
    # growth rate in the Bertram & Masel 2019 lottery model
    #
    # inputs:
    # b - juvenile birth rate
    # dOpt - death term of optimal genotype
    # sa - selection coefficient of beneficial mutations in "d" trait
    #
    # Output: 
    # iExt - largest fitness class after which there is negative growth in pop size
    # dMax - largest death term after which there is negative growth in pop size
    # di - list of death terms corresponding to the discrete absolute fit classes
    
    # Theoretical derivations show that maximum size of death term is given by 
    # value below. See Appendix A in manuscript.
    dMax = b+1
    
    # Recursively calculate set of absolute fitness classes 
    di = [dOpt]
    iMax = 0                            
    while (di[-1] < dMax):
        # loop until stop at dMax or greater reached
        di = di+[di[i-1]*(1+sa*(d[i-1]-1))]
        iMax = iMax + 1
    
    di = np.asarray(di)

    return [iExt,dMax,di]

#------------------------------------------------------------------------------
    
def get_c_selection_coefficient(b,y,cr,d_i):
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

    # calculate rate of increase in frequency per iteration
    r_rel = cr*(1-y)*(1-(1+b*y)*np.exp(-b*y))/(y+(1-y)*(1-np.exp(-b)))*(b*y-1+np.exp(-b*y))/(b*y*(1-np.exp(-b*y))+cr*(1-(1+b*y)*np.exp(-b*y)))
    
    # re-scale time to get rate of increase in frequency per generation
    tau_i = myfun.get_iterationsPerGeneration(d_i)
    
    sr = r_rel*tau_i
    
    return sr

#------------------------------------------------------------------------------
    
def get_iterationsPerGeneration(d_i):
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
    paramDict = dict([[paramList[i],paramValue[i]] for i in len(paramValue)])
    
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
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

# -----------------------------------------------------------------------------

def get_vDF_lnN_vers(N,s,U):
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
        
    v = s**2*(2*N + 2*np.log(s)-np.log(s/U))/(np.log(s/U)**2)
    
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
    
def get_extinction_class(b,d,sa):
    # function calculates the class for which population size has a negative 
    # growth rate in the Bertram & Masel 2019 lottery model
    #
    # inputs:
    # b - juvenile birth rate
    # d - minimum death rate of optimal genotype
    # sa - selection coefficient of beneficial mutations in "d" trait
    #
    # Output: 
    # i_ext - smallest absolute fitness class with negative growth
    #
    
    i_ext = int(np.ceil(np.log((b+1)/d)/np.log(1+sa)))
    
    return i_ext
#------------------------------------------------------------------------------

def get_class_death_rate(d,sa,i):
    # returns death rate of class i given parameters d,sa. Note need to have 
    # i <= ceiling(i_ext)
    #
    # inputs:
    # d - minimum death rate of optimal genotype
    # sa - selection coefficient of beneficial mutations in "d" trait
    # i - absolute fitness class
    #
    # Output: 
    # di - death rate of absolute fitness class i
    #
    
    di = d*((1+sa)**i)
    
    return di

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
    
def get_c_selection_coefficient(b,y,sr):
    # Calculate the "c" selection coefficient for the Bertram & Masel variable 
    # density lottery model for choice of ci = (1+sr)^i
    #
    # Inputs:
    # b - juvenile birth rate
    # y - equilibrium population density
    # sr - increase to ci from a single beneficial mutation is (1+sr)
    #
    # Output: 
    # eff_sr - selection coefficient of beneficial mutation in "c" trait
    #

    eff_sr = (sr*(1-y)*(1-(1+b*y)*np.exp(-b*y))/(y+(1-y)*(1-np.exp(-b)))) * ((b*y-1+np.exp(-b*y))/(b*y*(1-np.exp(-b*y))+sr*(1-(1+b*y)*np.exp(-b*y))))
    
    return eff_sr




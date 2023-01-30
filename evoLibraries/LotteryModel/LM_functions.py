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

import bisect
import csv
import math as math
import pickle 

from sympy import Matrix
from numpy.polynomial import Polynomial

from evoLibraries import constants as c 

#------------------------------------------------------------------------------
# Lottery Model Equations
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
        cbar = sum(m*c)/sum(m)
        
        # return the expected number of new adults
        return m * ( np.exp(-L) + ( calculate_Ri_term(m,c,U) + calculate_Ai_term(m,c,U) ) * (c/cbar) )
    
    else:
        # if the population has gone extinct
        return np.zeros(len(m))

#------------------------------------------------------------------------------
#   Lottery Model supporting calculations 
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

def get_d_SelectionCoeff(dWt,dMt):
    # Inputs:
    #- d1 wild type death term 
    #- d2 mutant type death term
    #
    # Outputs:
    #- selCoeff (rate of frequncy increase per generation)
    #
   
    selCoeff = (dWt-dMt)/(dMt*(dMt-1))
    
    return selCoeff    

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
    # get_iterationsPerGenotypeGeneration() calculates the number of iterations
    # corresponding to a generation for genotype with death term d_i in the 
    # Bertram & Masel Lotter Model.
    #
    # Inputs:
    # d_i - death term of genotype
    #
    # Output: 
    # tau_i - mean number of iterations per generation (i.e. average generation time in terms of iteration)
    #
    
    tau_i = 1/(d_i-1)
    
    return tau_i

#------------------------------------------------------------------------------
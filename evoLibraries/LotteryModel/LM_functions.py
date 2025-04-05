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

def get_eqPopDensity(b,d,option):
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
        return (1-y)*(1-np.exp(-b*y))-(d-1)*y
    
    # analytic approximation near extinction
    def eq_popsize_analytic_ext(b,d):
        # approximation near d ~ b+1 region
        eq_density = (b+2)/(2*b)*(1-np.sqrt(1-8*(b-d+1)/(b+2)**2))
        
        eq_density = np.max([eq_density,0]) # ensure density >= 0
        
        return eq_density
    
    # analytic approximation near d~dOpt (d evolution only)
    def eq_popsize_analytic_opt(b,d):
        
        y1 = (1-np.exp(-b)) / (d-np.exp(-b))  # 1st approximation
        
        eq_density = y1 + ( (1-np.exp(-b*y1)) - y1*(d-np.exp(-b*y1)) ) / \
                      ( (d-np.exp(-b*y1)) - (1-y1)*b*np.exp(-b*y1) )  # 2nd approximation
                                
        return eq_density
    
    
    if (d >= b+1):
        # if past extinction class, set density to zero
        eq_density = 0 
        
    else:
        # if among viable classes, calculate density 
        if option == 1:
            # approximation near optimal gentotype
            #eq_density = (1-np.exp(-b))/(di-np.exp(-b))+(di-1)/(di-np.exp(-b))* \
            #                (np.exp(-b)-np.exp(-b*(1-np.exp(-b))/(di-np.exp(-b))))/ \
            #                        (di-np.exp(-b*(1-np.exp(-b))/(di-np.exp(-b))))
            eq_density = eq_popsize_analytic_opt(b,d)
            eq_density = np.max([eq_density,0]) # ensure density >= 0
            
        elif option == 2:
            # approximation near extinction genotype
            eq_density = eq_popsize_analytic_ext(b,d)
            eq_density = np.max([eq_density,0]) # ensure density >= 0
            
        else:
            # numerical solution to steady state population size equation
            eq_density = opt.broyden1(eq_popsize_err,[1], f_tol=1e-14)
            eq_density = np.max([eq_density[0],0]) # ensure density >= 0
        
    # now do some numerical checks to get the best possible approximation 
    # near the extinction class, i.e. if we numerically get zero, then use
    # analytic approximation (option 2).
    if ((d < b + 1) and (eq_density == 0)):
        eq_density = eq_popsize_analytic_ext(b,d)
        
    return eq_density

#------------------------------------------------------------------------------

def get_eqPopDensity_bEvoAnalytic(b,d):
    # variable density lottery model analytic approximation that relates 
    # equilibrium density and b-values with a fixed d-value. The aproximation
    # 
    # Inputs:
    # b - juvenile birth term
    # b0 - max 
    # d - death term of abs-fitness class i
    #
    # Output: 
    # eq_density - equilibrium population density
    #
    
    # define minimum b value
    b0 = d-1
    
    # calculate equilibrium density
    eq_density = ( 1-np.exp(-(b-b0)/d) ) / ( d-np.exp(-(b-b0)/d) )
    
    return eq_density
    
#------------------------------------------------------------------------------

def get_d_SelectionCoeff(dWt,dMt):
    # Calculate the "d" selection coefficient for the Bertram & Masel variable 
    # density lottery model
    #
    # Inputs:
    #- dWt wild type death term 
    #- dMt mutant type death term
    #
    # Outputs:
    #- sd (rate of frequency increase per generation)
    #
    
    r_abs = (dWt-dMt)/dMt    # more numerically accurate than dWt/dMt-1
    
    # get time-scale of a generation
    tau = get_iterationsPerGenotypeGeneration(dMt)
    
    # re-scale time to get rate of increase in frequency per generation
    sd = r_abs * tau
    
    return sd

#------------------------------------------------------------------------------

def get_b_SelectionCoeff(bWt,bMt,d):
    # Calculate the "d" selection coefficient for the Bertram & Masel variable 
    # density lottery model
    #
    # Inputs:
    #- bWt wild type death term 
    #- bMt mutant type death term
    #
    # Outputs:
    #- sb (rate of frequency increase per generation)
    #
    
    # b increment size
    del_b = bMt-bWt
    
    # rate of frequency increase per time iteration
    r_abs = (d-1)*del_b/(bWt*d)
    
    # get time-scale of a generation
    tau = get_iterationsPerGenotypeGeneration(d)
    
    # re-scale time to get rate of increase in frequency per generation
    sb = r_abs * tau
    
    return sb

#------------------------------------------------------------------------------
    
def get_c_SelectionCoeff(b,y,cr,dMt):
    # Calculate the "c" selection coefficient for the Bertram & Masel variable 
    # density lottery model for choice of ci = (1+c+)^i. We use the multiplicative
    # model of ci increments because the Bertram & Masel variable density lottery
    # model is discrete.
    # 
    #
    # Inputs:
    # b - juvenile birth rate
    # y - equilibrium population density
    # cr - increase to ci from a single beneficial mutation is (1+cr)
    # dMt - death term of the mutant class
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
    
    # get time-scale of a generation
    tau = get_iterationsPerGenotypeGeneration(dMt)
    
    # re-scale time to get rate of increase in frequency per generation
    sr = r_rel * tau
    
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

def run_lotteryModelSelection_noMutations(nij,bij,dij,cij,T,tmax,outpar):
    # runs the deterministic selection dynamics of the Lotter Model tmax iterations
    # i.e. no mutations, only selection dynamics. This is used to test analytic
    # approdiximations of paper.
    # 
    # inputs:
    # nij, bij, dij, cij - m-by-n arrays assocated with travelinging wave model
    #     NOTE: Abs & Rel fitness dimensions increment by row & cols, respectively 
    # T -  territory size
    # outputFileParameters - 
    #     0. output file to capture snapshots (string with file full path)
    #     1. number of iterations per snapshot (integer value)
    
    n = nij.flatten()
    b = bij.flatten()
    d = dij.flatten()
    c = cij.flatten()
    
    nclass = len(n)
    kk = np.array([ii for ii in range(nclass)]) 
    
    filepath = outpar[0]
    tsample = outpar[1]
    
    t = 0
    
    while (t < tmax):
        # get pop size and number of unoccupied territories
        Ntot  = np.sum(n)
        Uterr = T-Ntot
        
        m = n*b*(Uterr/T)
        
        delta_n = deltnplus(m,c,Uterr)
        n = (n + delta_n)/d
        
        out = removeExtinctClasses(n,b,d,c,kk)
        
        n = out[0]
        b = out[1]
        d = out[2]
        c = out[3]
        kk = out[4]
        
        if (np.mod(t,tsample)==0):
            printPopDetails(t, n, kk, nclass, filepath)
        
        t = t+1
        
    return None

#------------------------------------------------------------------------------

def removeExtinctClasses(n,b,d,c,k):
    # method that removes extinct clases
    
    ntemp = []
    btemp = []
    dtemp = []
    ctemp = []
    ktemp = []
    
    for ii in range(len(n)):
        if (n[ii] >= 1):
            ntemp.append(n[ii])
            btemp.append(b[ii])
            dtemp.append(d[ii])
            ctemp.append(c[ii])
            ktemp.append(k[ii])
    
    ntemp = np.asarray(ntemp)
    btemp = np.asarray(btemp)
    dtemp = np.asarray(dtemp)
    ctemp = np.asarray(ctemp)
        
    return [ntemp,btemp,dtemp,ctemp,ktemp]

# -----------------------------------------------------------------------------

def printPopDetails(t,n,kk,nclass,outfile):
    # simple function to print outputs from lottery model selection dynamics
    
    datatemp = np.zeros(nclass+1)
    datatemp[0]
    
    for ii in range(len(kk)):
        datatemp[kk[ii]+1] = n[kk[ii]]
        
    
    with open(outfile, "a") as file:
        # output data collected
        file.write((','.join(tuple(['%f']*len(datatemp)))+'\n') % tuple(datatemp))
    
    return None

#------------------------------------------------------------------------------

def get_flatArryMap_1Dto2D(kkVal,arryShape):
    # simple utility that maps index value in 1d array to 2d form
    # arryShape = (n-rows,n-cols)
    nRow = arryShape[0]
    nCol = arryShape[1]
    
    ijVal = [0,0]
    ijVal[1] = int(np.mod(kkVal,nCol))
    ijVal[0] = int((kkVal-ijVal[1])/nRow )
    
    return ijVal

#------------------------------------------------------------------------------

def get_flatArryMap_2Dto1D(ijVal,arryShape):
    # simiple utility that maps index value in 2d array to 1d form
    # arryShape = (n-rows,n-cols), ijVal is list with ii and jj as entries
    nRow = arryShape[0]
    nCol = arryShape[1]
    
    kkVal = int(nRow*ijVal+ijVal[1])
    
    return kkVal
    
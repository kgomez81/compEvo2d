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

import csv
import pickle 

from numpy.polynomial import Polynomial

# *****************************************************************************
# Markov Chain Class
# *****************************************************************************



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

def get_absoluteFitnessClassesDRE(b,dOpt,alpha):
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
    
    iStop = True
    
    while (iStop):

        # get next d-term
        dNext = dMax*(dOpt/dMax)**st.logser.cdf(ii,alpha)            
        
        selCoeff_ii = get_d_SelectionCoeff(di[-1],dNext) 
        
        if (selCoeff_ii >1e-15):
            # loop until stop at dStop or greater reached
            di = di + [ dNext ]
            ii = ii + 1
        else:
            # size of selection coefficient fell below numerical precision
            # so just end the loop
            iStop = False
    
    di = np.asarray(di)
    iMax = int(di.shape[0]-1)
    
    return [dMax,di,iMax]

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

def rescale_v_iterTimeScale(di,v):
    v_rescaled = []
    
    for ii in range(len(v)):     
        v_rescaled = v_rescaled + [v[ii]*(di[ii+1]-1)]
    
    return v_rescaled

#------------------------------------------------------------------------------

def get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i, N_i):
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
    Npop = N_i[idxMin]

    # calculate the regime IDs for each trait
    #  0: Bad evo parameters
    #  1: successional
    #  2: multiple mutations
    #  3: diffusion
    # -1: regime undetermined, i.e. in transition region   
        
    regimeID_a = get_regimeID(Npop,sa,Ua,sa)
    regimeID_r = get_regimeID(Npop,sr,Ur,sr)

    
    # calculate the appropriate rho
    if (regimeID_a == 1) or (regimeID_r == 1):
        # either or both in successional regime, no clonal interference
        rho = 0
    
    elif (regimeID_a == 2) and (regimeID_r == 2):
        # both traits in multiple mutations regime
        rho = (sr/np.log(sr/Ur))**2 / (sa/np.log(sa/Ua))**2
        
    elif (regimeID_a == 3) and (regimeID_r == 2):
        # abs trait in diffusion and rel trait in multiple mutations regime
        Da = 0.5*Ua*sa**2
        
        rho = (sr/np.log(sr/Ur))**2 / (Da**(2.0/3.0)/(3*np.log(Da**(1.0/3.0)*Npop)**(2.0/3.0)))               
        
    elif (regimeID_a == 2) and (regimeID_r == 3):
        # rel trait in diffusion and abs trait in multiple mutations regime
        Dr = 0.5*Ur*sr**2
        
        rho = (Dr**(2.0/3.0)/(3*np.log(Dr**(1.0/3.0)*Npop)**(2.0/3.0))) / (sa/np.log(sa/Ua))**2
        
    elif (regimeID_a == 3) and (regimeID_r == 3):
        # both traits in diffusion
        Da = 0.5*Ua*sa**2
        Dr = 0.5*Ur*sr**2
        
        rho = (Dr**(2.0/3.0)/(3*np.log(Dr**(1.0/3.0)*Npop)**(2.0/3.0))) / (Da**(2.0/3.0)/(3*np.log(Da**(1.0/3.0)*Npop)**(2.0/3.0)))
        
    else:
        rho = np.nan
            
    return [rho, sa, Ua, sr, Ur]

#------------------------------------------------------------------------------

def get_intersection_popDensity(va_i, vr_i, eq_yi):
    # function to calculate the intersection equilibrium density
        
    # find index that minimizes |va-vr| but exclude extinction class (:-1)
    idxMin = np.argmin(np.abs(np.asarray(va_i[0:-1])-np.asarray(vr_i[0:-1])))
    
    # Definition of the gamma at intersection in paper
    yiInt = eq_yi[idxMin]
    
    return yiInt

#------------------------------------------------------------------------------
    
def get_contourPlot_arrayData(myOptions):
    # Generic function takes the provided options and generates data needed to
    # creat contour plot of rho and gamma.
    
    # set values of first parameter
    varParam1A = myOptions.varNames[0][0]
    varParam2A = myOptions.varNames[0][1]
    
    varParam1B = myOptions.varNames[1][0]
    varParam2B = myOptions.varNames[1][1]
    
    x1LwrBnd_log10 = np.log10(myOptions.varBounds[0][0]*myOptions.params[varParam1A])
    x1UprBnd_log10 = np.log10(myOptions.varBounds[0][1]*myOptions.params[varParam1A])
    if myOptions.modelType == 'RM':
        x1RefVal_log10 = np.log10(myOptions.params[varParam1B])
    else:
        alpha = myOptions.params[varParam1B]
        de = myOptions.params['b']+1
        d0 = myOptions.params['dOpt']
        sa1_mid = 0.5*(1-alpha)*(de-d0)/((de+(d0-de)*(1-alpha))*(de+(d0-de)*(1-alpha)-1))
        x1RefVal_log10 = np.log10(sa1_mid)
    
    X1_vals = np.logspace(x1LwrBnd_log10, x1UprBnd_log10, num=myOptions.varBounds[0][2])
    X1_ref  = np.logspace(x1RefVal_log10, x1RefVal_log10, num=1                        )
    
    x2LwrBnd_log10 = np.log10(myOptions.varBounds[1][0]*myOptions.params[varParam2A])
    x2UprBnd_log10 = np.log10(myOptions.varBounds[1][1]*myOptions.params[varParam2A])
    x2RefVal_log10 = np.log10(myOptions.params[varParam2B])
    
    X2_vals = np.logspace(x2LwrBnd_log10, x2UprBnd_log10, num=myOptions.varBounds[1][2])
    X2_ref  = np.logspace(x2RefVal_log10, x2RefVal_log10, num=1                        )
    
    X1_ARRY, X2_ARRY = np.meshgrid(X1_vals, X2_vals)
    RHO_ARRY = np.zeros(X1_ARRY.shape)
    Y_ARRY = np.zeros(X1_ARRY.shape)

    # arrays to store effecttive s and U values
    effSa_ARRY = np.zeros(X1_ARRY.shape)
    effUa_ARRY = np.zeros(X1_ARRY.shape)

    effSr_ARRY = np.zeros(X1_ARRY.shape)
    effUr_ARRY = np.zeros(X1_ARRY.shape)
    
    paramsTemp = cpy.copy(myOptions.params)

    # --------------------------------------------------------------------------
    # Calculated rho values for T vs 2nd parameter variable
    # --------------------------------------------------------------------------
    
    for ii in range(int(X1_ARRY.shape[0])):
        for jj in range(int(X2_ARRY.shape[1])):
            
            # set cr and sa values (selection coefficient)
            paramsTemp[varParam1A] = X1_ARRY[ii,jj]
            paramsTemp[varParam1B] = (myOptions.params[varParam1B],X1_ref[0])[myOptions.modelType == 'RM']
            
            # set Ua values and Ur values (mutation coefficient)
            paramsTemp[varParam2A] = X2_ARRY[ii,jj]
            paramsTemp[varParam2B] = X2_ref[0]
            
            # Calculate absolute fitness state space. 
            if myOptions.modelType == 'RM':
                [dMax,di,iMax] = get_absoluteFitnessClasses(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['sa'])
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = get_MChainPopParameters(paramsTemp,di,iMax,myOptions.yi_option)        
            else:
                iStop = np.log(0.01)/np.log(myOptions.params['alpha'])-1  # stop at i steps to get di within 5% of d0, i.e. (di-d0)/(dMax-d0) = 0.05.
                [dMax,di,iMax] = get_absoluteFitnessClassesDRE(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['alpha'],iStop)
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = get_MChainPopParametersDRE(paramsTemp,di,iMax,myOptions.yi_option)
                
            pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
            pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])
            
            # Use s values for pFix until we get sim pFix values can be obtained
            if myOptions.modelType == 'RM':
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                                get_MChainEvoParameters(paramsTemp,di,iMax,pFixAbs_i,pFixRel_i,myOptions.yi_option)
            else:
                [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                                get_MChainEvoParametersDRE(paramsTemp,di,iMax,pFixAbs_i,pFixRel_i,myOptions.yi_option)                                    
                                
            [RHO_ARRY[ii,jj], effSa_ARRY[ii,jj], effUa_ARRY[ii,jj], effSr_ARRY[ii,jj], effUr_ARRY[ii,jj] ] = \
                                get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i,eq_Ni)   
                            
            Y_ARRY[ii,jj] = get_intersection_popDensity(va_i, vr_i, eq_yi)   
    
    
    with open(myOptions.saveDataName, 'wb') as f:
        pickle.dump([X1_ARRY,X2_ARRY,RHO_ARRY,Y_ARRY,X1_ref,X2_ref, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp,dMax], f)
    
    return None














# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import evo_library as myfun            # my functions in a seperate file

# --------------------------------------------------------------------------
#                               Parameters
# --------------------------------------------------------------------------
# load parameters from a csv file. The parameters must be provided in the 
# following order (numeric values). All input csv files must be available in 
# inputs or outputs directory. Below are examples of parameters that should be 
# given in the csv file.
#
#T   = 1e9          # carrying capacity
#b   = 2.0          # birth rate
#do  = 100/98.0     # minimum death rate / death rate of optimal genotype
#sa  = 1e-2         # selection coefficient of beneficial mutation in 
#Ua  = 1e-5         # max beneficial mutation rate in trait "d"
#Uad = 1e-5         # deleterious mutation rate in trait "d"
#cr  = 0.175        # increment to "c" is (1+cr)
#Ur  = 1e-5         # beneficial mutation rate in trait "c"
#Urd = 1e-5         # deleterious mutation rate in trait "c"
#R  = 1/130.0      # rate of environmental change per iteration
#
# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp01_parameters_VaVeIntersect.csv'
paramList = ['T','b','dOpt','sa','UaMax','Uad','cr','Ur','Urd','R']
params = myfun.read_parameterFile(paramFile,paramList)

# --------------------------------------------------------------------------
#                       Markov Evolution Parameters
# --------------------------------------------------------------------------
# Calculate absolute fitness state space. This requires specificying:
# dMax  - max size of death term that permits non-negative growth in abundances
# di    - complete of death terms of the various absolute fitness states
[dMax,di] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

# set index of extinction class
iExt = int(di.shape[0]-1)

# pFix values from simulations are loaded for abs-fit mutations to states 0,...,iMax-1 
pFixAbs_File = 'outputs/evoExp01_absPfix.csv'
pFixAbs_i     = myfun.read_pFixOutputs(pFixAbs_File,iExt)

# pFix values from simulations are loaded for rel-fit mutations at states 1,...,iMax 
pFixRel_File = 'outputs/evoExp01_relPfix.csv'
pFixRel_i    = myfun.read_pFixOutputs(pFixRel_File,iExt)

# set root solving option for equilibrium densities
# (1) low-density analytic approximation 
# (2) high-density analytic approximation
# (3) root solving numerical approximation
yi_option = 3  

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
    eq_yi   = eq_yi + [myfun.get_eqPopDensity(params['b'],di[ii],yi_option)]
    eq_Ni   = eq_Ni + [params['T']*eq_yi[-1]]
    sr_i    = sr_i + [myfun.get_c_SelectionCoeff(params['b'],eq_yi[-1],params['cr'],di[ii])]
    sa_i    = sa_i + [params['sa']]
    
    # rates of fitness change ( on time scale of generations)
    va_i = va_i + [myfun.get_rateOfAdapt(eq_Ni[-1],sa_i[-1],Ua_i[-1],pFixAbs_i[ii-1][0])]
    vr_i = vr_i + [myfun.get_rateOfAdapt(eq_Ni[-1],sr_i[-1],Ur_i[-1],pFixRel_i[ii-1][0])]
    ve_i = ve_i + [sa_i[-1]*params['R']/(di[ii]-1)]  
    
# convert all list into arrays for plotting purposes
state_i = np.asarray(state_i)
Ua_i    = np.asarray(Ua_i)
Ur_i    = np.asarray(Ur_i)
eq_yi   = np.asarray(eq_yi)
eq_Ni   = np.asarray(eq_Ni)
sr_i    = np.asarray(sr_i)
sa_i    = np.asarray(sa_i)

va_i    = np.asarray(va_i)
vr_i    = np.asarray(vr_i)
ve_i    = np.asarray(ve_i)

# --------------------------------------------------------------------------
#                               Figure 
# --------------------------------------------------------------------------
fig1,ax1 = plt.subplots(1,1,figsize=[7,6])
ax1.scatter(state_i,va_i,color="blue",linewidth=1.0,label=r'$v_a$')
ax1.scatter(state_i,vr_i,color="orange",linewidth=1.0,label=r'$v_r$')
ax1.plot(state_i,ve_i,color="black",linewidth=5,label=r'$v_e$')

# axes and label adjustements
ax1.set_xlim(-iExt-1,0)
ax1.set_ylim(0,2.52e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])
ax1.set_xticks([-25*i for i in range(0,iExt/25+1)])
ax1.set_xticklabels([str(25*i) for i in range(0,iExt/25+1)],fontsize=16)
ax1.set_yticks([1e-5*5*i for i in range(0,6)])
ax1.set_yticklabels([str(5*i/10.0) for i in range(0,6)],fontsize=16)
ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
ax1.legend(fontsize = 14,ncol=1,loc='lower right')

# annotations
ax1.plot([-88,-88],[0,1.6e-4],c="black",linewidth=2,linestyle='--')
ax1.annotate("", xy=(-89,0.7e-4), xytext=(-104,0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
ax1.annotate("", xy=(-87,0.7e-4), xytext=(-72, 0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4})
plt.text(-84,0.29e-4,r'$i^*=88$',fontsize = 18)
plt.text(-84,0.10e-4,r'$i_{ext}=180$',fontsize = 18)
plt.text(-190,2.58e-4,r'$\times 10^{-4}$', fontsize = 20)
plt.text(-175,2.30e-4,r'(A)', fontsize = 22)

# save figures
#plt.tight_layout()
fig1.savefig('figures/fig_MChain_VaVeIntersection.pdf')
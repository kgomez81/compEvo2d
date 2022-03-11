# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:39:02 2022

@author: dirge
"""


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
# Calculate Markov Chain Evolution Parameters - Panel (A)
# --------------------------------------------------------------------------
# load parameters from a csv file. The parameters must be provided in the 
# following order (numeric values). All input csv files must be available in 
# inputs or outputs directory. Below are examples of parameters that should be 
# given in the csv file.
#
# T   = 1e9          # carrying capacity
# b   = 2.0          # birth rate
# do  = 100/98.0     # minimum death rate / death rate of optimal genotype
# sa  = 1e-2         # selection coefficient of beneficial mutation in 
# Ua  = 1e-5         # max beneficial mutation rate in trait "d"
# Uad = 1e-5         # deleterious mutation rate in trait "d"
# cr  = 0.175        # increment to "c" is (1+cr)
# Ur  = 1e-5         # beneficial mutation rate in trait "c"
# Urd = 1e-5         # deleterious mutation rate in trait "c"
# R  = 1/130.0      # rate of environmental change per iteration
#

# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp02_parameters_VaVrIntersect.csv'
params = myfun.read_parameterFile(paramFile)

# Calculate absolute fitness state space. This requires specificying:
# dMax  - max size of death term that permits non-negative growth in abundances
# di    - complete of death terms of the various absolute fitness states
# iExt  - extinction class
[dMax,di,iExt] = myfun.get_absoluteFitnessClasses(params['b'],params['dOpt'],params['sa'])

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

[state1_i,Ua1_i,Ur1_i,eq_y1i,eq_N1i,sr1_i,sa1_i,va1_i,vr1_i,ve1_i] = \
                    myfun.get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option)

# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp04_parameters_VaVrIntersect.csv'
params = myfun.read_parameterFile(paramFile)

[state2_i,Ua2_i,Ur2_i,eq_y2i,eq_N2i,sr2_i,sa2_i,va2_i,vr2_i,ve2_i] = \
                    myfun.get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option)

# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp05_parameters_VaVrIntersect.csv'
params = myfun.read_parameterFile(paramFile)

[state3_i,Ua3_i,Ur3_i,eq_y3i,eq_N3i,sr3_i,sa3_i,va3_i,vr3_i,ve3_i] = \
                    myfun.get_MChainEvoParameters(params,di,iExt,pFixAbs_i,pFixRel_i,yi_option)

# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, ax1 = plt.subplots(1,1,figsize=[7,6])

#ax1.plot(state1_i,ve1_i,color="black",linewidth=2,label=r'$v_e$')
#ax1.plot(eq_y1i,va1_i,color="green",linewidth=2,label=r'$v_{a,1}$')
#ax1.plot(eq_y1i,vr1_i,'-.',color="green",linewidth=2,label=r'$v_{r,1}$')

ax1.plot(eq_y2i,va2_i,color="blue",linewidth=2,label=r'$v_a (T=10^5)$')
ax1.plot(eq_y2i,vr2_i,'-.',color="blue",linewidth=2,label=r'$v_r (T=10^5)$')
ax1.plot(eq_y3i,va3_i,color="cyan",linewidth=2,label=r'$v_a (T=10^7)$')
ax1.plot(eq_y3i,vr3_i,'-.',color="cyan",linewidth=2,label=r'$v_r (T=10^7)$')

ax1.scatter([0.8805],[1.0223e-4],marker="o",s=40,c="black")
ax1.scatter([0.8923],[0.492e-4],marker="o",s=40,c="black")
ax1.plot([0.8805,0.8923],[1.0223e-4,0.492e-4],c="black",linewidth=2,linestyle='-')

# axes and label adjustements
ax1.set_xlim(0.6,1)
#ax1.set_xticks([-25*i for i in range(0,iExt/25+1)])
#ax1.set_xticklabels([str(25*i) for i in range(0,iExt/25+1)],fontsize=16)
#ax1.set_xticklabels(["" for i in range(0,iExt/25+1)],fontsize=16)
#ax1.set_yticks([0.2e-4*i for i in range(0,7)])
#ax1.set_yticklabels([str(2*i/10.0) for i in range(0,7)],fontsize=16)
ax1.set_ylim(0,1.3e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])
ax1.set_xlabel(r'Equilibrium Population Density',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
#ax1.legend(fontsize = 14,ncol=1,loc='lower right')

# save figure
plt.tight_layout()
fig1.savefig('figures/fig_MChain_VaVrIntersectionVariableT.pdf')
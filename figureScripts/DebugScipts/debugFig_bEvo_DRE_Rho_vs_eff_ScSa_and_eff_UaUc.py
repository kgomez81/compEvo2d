# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:59:26 2023

@author: kgomez

This script was generate to test the MC array class results
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

# NOTE: to use this script, you need to have run one of the MC array figure scripts

import matplotlib.pyplot as plt
import numpy as np
import time

import os
import sys

from evoLibraries.MarkovChain import MC_array_class as mcArry
# from evoLibraries.MarkovChain import MC_functions as mcFun
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

#%% Final section to check individual entries of rho plots MC models

# ii=8
# jj=6

ii=9
jj=10

figSelect = 2

mcTestEvoOptions = mcModels.get_params_ij(ii,jj)
mcTestModel = mcDRE.mcEvoModel_DRE(mcTestEvoOptions)

# mcTestEqParams = mcTestModel.get_stable_state_evo_parameters()
# # testParamGrid = mcModels.get_evoParam_grid('UdMax',0)

# fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[7,12])
# ax1.scatter(mcTestModel.state_i,np.log10(mcTestModel.vd_i),label='vd')
# ax1.scatter(mcTestModel.state_i,np.log10(mcTestModel.vc_i),label='vc')
# ax1.legend()

# params_stable_state = mcTestModel.get_stable_state_evo_parameters()
# rho_val = mcTestModel.calculate_evoRho()

# ax1.text(-120,-1,params_stable_state['eqState'], fontsize = 22)
# ax1.text(-120,-2,rho_val, fontsize = 22)

# ax2.scatter(mcTestModel.state_i,np.log10(mcTestModel.sd_i),label='sd')
# ax2.scatter(mcTestModel.state_i,np.log10(mcTestModel.pFix_d_i),label='pFix_d')
# ax2.legend()
# ax3.scatter(mcTestModel.state_i,np.log10(mcTestModel.sc_i),label='sc')
# ax3.scatter(mcTestModel.state_i,np.log10(mcTestModel.pFix_c_i),label='pFix_c')
# ax3.legend()

if figSelect == 1:
    # figures by state
    fig1,((ax11,ax12,ax13,ax14),(ax21,ax22,ax23,ax24)) = plt.subplots(2,4,figsize=[30,17])
    ax11.scatter(mcTestModel.state_i,mcTestModel.va_i,label='va')
    ax11.scatter(mcTestModel.state_i,mcTestModel.vc_i,label='vc')
    ax11.set_xlabel('MC state')
    ax11.set_ylabel('RoA (v)')
    ymax = max([max(mcTestModel.va_i),max(mcTestModel.vc_i)])
    params_stable_state = mcTestModel.get_stable_state_evo_parameters()
    ax11.text(0,0.5*ymax,params_stable_state['eqState'], fontsize = 22)
    ax11.text(0,0.75*ymax,mcTestModel.calculate_evoRho(), fontsize = 22)
    ax11.legend()
    ax21.scatter(mcTestModel.state_i,mcTestModel.evoRegime_a_i,label='reg_a')
    ax21.scatter(mcTestModel.state_i,mcTestModel.evoRegime_c_i,label='reg_c')
    ax21.set_xlabel('MC state')
    ax21.set_ylabel('v-Regime (0-?,1-S,2-MM,3-D)')
    ax21.legend()
    
    ax12.scatter(mcTestModel.state_i,mcTestModel.pFix_a_i,label='pFix_a')
    ax12.scatter(mcTestModel.state_i,mcTestModel.pFix_c_i,label='pFix_c')
    ax12.set_xlabel('MC state')
    ax12.set_ylabel('pfix')
    ax12.legend()
    ax22.scatter(mcTestModel.state_i,mcTestModel.sa_i,label='sa')
    ax22.scatter(mcTestModel.state_i,mcTestModel.sc_i,label='sc')
    ax22.set_xlabel('MC state')
    ax22.set_ylabel('selection coeff.')
    ax22.legend()
    
    ax13.scatter(mcTestModel.state_i,mcTestModel.eq_yi,label='eq_yi')
    ax13.set_xlabel('MC state')
    ax13.set_ylabel('eq_yi')
    ax13.legend()
    ax23.scatter(mcTestModel.eq_yi,mcTestModel.pFix_a_i,label='pfix_a vs eq_yi')
    ax23.scatter(mcTestModel.eq_yi,mcTestModel.pFix_c_i,label='pFix_c vs eq_yi')
    ax23.set_xlabel('eq_yi')
    ax23.set_ylabel('pFix')
    ax23.legend()
    
    ax14.scatter(mcTestModel.pFix_a_i,mcTestModel.va_i,label='va vs pFix')
    ax14.scatter(mcTestModel.pFix_c_i,mcTestModel.vc_i,label='vc vs pFix')
    ax14.set_xlabel('pFix')
    ax14.set_ylabel('RoA (v)')
    ax14.legend()
    ax24.scatter(mcTestModel.sa_i,mcTestModel.pFix_a_i,label='pfix_a vs sa')
    ax24.scatter(mcTestModel.sc_i,mcTestModel.pFix_c_i,label='pFix_c vs sc')
    ax24.plot([0,max([max(mcTestModel.sa_i),max(mcTestModel.sc_i)])],[0,max([max(mcTestModel.pFix_a_i),max(mcTestModel.pFix_c_i)])],c='black',label='y=x')
    ax24.set_xlabel('selection coeff')
    ax24.set_ylabel('pFix')
    ax24.legend()
    
if figSelect == 2:
    # figures by bi
    fig2,((ax11,ax12,ax13,ax14),(ax21,ax22,ax23,ax24)) = plt.subplots(2,4,figsize=[30,17])
    ax11.scatter(mcTestModel.bi,mcTestModel.va_i,label='va')
    ax11.scatter(mcTestModel.bi,mcTestModel.vc_i,label='vc')
    ax11.set_xlabel('b_i')
    ax11.set_ylabel('RoA (v)')
    ymax = max([max(mcTestModel.va_i),max(mcTestModel.vc_i)])
    params_stable_state = mcTestModel.get_stable_state_evo_parameters()
    ax11.text(min(mcTestModel.bi),0.75*ymax,mcTestModel.calculate_evoRho(), fontsize = 22)
    ax11.legend()
    ax21.scatter(mcTestModel.bi,mcTestModel.evoRegime_a_i,label='reg_a')
    ax21.scatter(mcTestModel.bi,mcTestModel.evoRegime_c_i,label='reg_c')
    ax21.set_xlabel('b_i')
    ax21.set_ylabel('v-Regime (0-?,1-S,2-MM,3-D)')
    ax21.legend()
    
    ax12.scatter(mcTestModel.bi,mcTestModel.pFix_a_i,label='pFix_a')
    ax12.scatter(mcTestModel.bi,mcTestModel.pFix_c_i,label='pFix_c')
    ax12.set_xlabel('b_i')
    ax12.set_ylabel('pfix')
    ax12.legend()
    ax22.scatter(mcTestModel.bi,mcTestModel.sa_i,label='sa')
    ax22.scatter(mcTestModel.bi,mcTestModel.sc_i,label='sc')
    ax22.set_xlabel('b_i')
    ax22.set_ylabel('selection coeff.')
    ax22.legend()
    
    ax13.scatter(mcTestModel.bi,mcTestModel.eq_yi,label='eq_yi')
    ax13.set_xlabel('b_i')
    ax13.set_ylabel('eq_yi')
    ax13.legend()
    ax23.scatter(mcTestModel.eq_yi,mcTestModel.pFix_a_i,label='pfix_a vs eq_yi')
    ax23.scatter(mcTestModel.eq_yi,mcTestModel.pFix_c_i,label='pFix_c vs eq_yi')
    ax23.set_xlabel('eq_yi')
    ax23.set_ylabel('pFix')
    ax23.legend()
    
    ax14.scatter(mcTestModel.pFix_a_i,mcTestModel.va_i,label='va vs pFix')
    ax14.scatter(mcTestModel.pFix_c_i,mcTestModel.vc_i,label='vc vs pFix')
    ax14.set_xlabel('pFix')
    ax14.set_ylabel('RoA (v)')
    ax14.legend()
    
    ax24.scatter(mcTestModel.sa_i,mcTestModel.pFix_a_i,label='pfix_a vs sa')
    ax24.scatter(mcTestModel.sc_i,mcTestModel.pFix_c_i,label='pFix_c vs sc')
    ax24.plot([0,max([max(mcTestModel.sa_i),max(mcTestModel.sc_i)])],[0,max([max(mcTestModel.sa_i),max(mcTestModel.sc_i)])],c='black',label='y=x')
    ax24.set_xlabel('selection coeff')
    ax24.set_ylabel('pFix')
    ax24.set_ylim([0,max(mcTestModel.sc_i)])
    ax24.legend()

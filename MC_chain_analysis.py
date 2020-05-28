# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020

@author: Kevin Gomez
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import fig_functions as myfun            # my functions in a seperate file

# reload(myfun)
# parameters
T = 1e9         # carrying capacity
b=2.0           # juvenile birth rate
d=100/98.0      # minimum death rate / death rate of optimal genotype
R=1/130.0       # rate of environmental change

sa = 1e-2       # selection coefficient of beneficial mutation in 
Ua = 1e-6       # beneficial mutation rate in trait "d"
Uad = 1e-5      # deleterious mutation rate in trait "d"

Ur=1e-5         # beneficial mutation rate in trait "c"
Urd=1e-5        # deleterious mutation rate in trait "c"
sr = 0.175     # multiplicative increment to "c" is (1+sr)

yi_option = 1   # select option to get equilibrium population density (=1,2,3)

abs_fit_clss = []   # absolute fitness classes
Ua_i = []       # absolute fitness mutation rate
eq_yi = []      # equilibrium density of fitness class i
eq_Ni = []      # equilibrium population size of fitness class i
eff_sr_i = []   # selection coefficient of "c" trait beneficial mutation
eff_sa_i = []   # selection coefficient of "c" trait beneficial mutation

va_i = []       # rate of adaptation in absolute fitness trait alone
vr_i = []       # rate of adaptation in relative fitness trait alone
ve_i = []       # rate of fitness decrease due to environmental degradation

prob_incr_abs = []    # transition probability i->i-1
prob_decr_abs = []    # transition probability i->i+1

i_ext = myfun.get_extinction_class(b,d,sa)

for i in range(-i_ext+1,0):
    # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
    Uai = -i*Ua
    eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
    eqNi = T*eqyi
    effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
    effsai = sa*myfun.get_class_death_rate(d,sa,-i+1)
    
    # rates of fitness change (relative scale)
    vai = myfun.get_vDF(eqNi,effsai,Uai)
    vri = myfun.get_vDF(eqNi,effsri,Ur)
    vei = sa*R
    
    abs_fit_clss = abs_fit_clss + [i]
    Ua_i = Ua_i + [Uai]
    eq_yi = eq_yi + [eqyi]
    eq_Ni = eq_Ni + [eqNi]
    eff_sr_i = eff_sr_i + [effsri]
    eff_sa_i = eff_sa_i + [effsai]
    va_i = va_i + [vai]
    vr_i = vr_i + [vri]
    ve_i = ve_i + [vei]
    
    prob_incr_abs = prob_incr_abs +[(vai/sa)]
    prob_decr_abs = prob_decr_abs +[R]
    
# state space
abs_fit_clss = np.asarray(abs_fit_clss)
Ua_i = np.asarray(Ua_i)
eq_yi = np.asarray(eq_yi)
eq_Ni = np.asarray(eq_Ni)
eff_sr_i = np.asarray(eff_sr_i)
eff_sa_i = np.asarray(eff_sa_i)
va_i = np.asarray(va_i)
vr_i = np.asarray(vr_i)

# memoryless transition probabilities
prob_incr_abs = np.asarray(prob_incr_abs)
prob_decr_abs = np.asarray(prob_decr_abs)

coeff_dT_a = (eff_sa_i/(np.log(eff_sa_i/Ua_i)))**2
coeff_dT_r = (eff_sr_i/(np.log(eff_sr_i/Ur)))**2

# ------------------ figure of Markov chain approximation --------------------

fig1,ax1 = plt.subplots(1,1,figsize=[7,7])
ax1.scatter(abs_fit_clss,va_i,color="blue",linewidth=2,label=r'$v_a$')
ax1.scatter(abs_fit_clss,vr_i,color="red",linewidth=2,label=r'$v_r$')
ax1.plot(abs_fit_clss,ve_i,color="green",linewidth=2,label="Env=1/130")
ax1.set_ylim(0,np.max(1.25*va_i))
ax1.set_xlabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
plt.text(-20,1.2e-5 ,r'$i_{max} = -98$',fontsize=16)
plt.text(-20,0.5e-5 ,r'$i_{ext} \approx -64$',fontsize=16)
plt.tight_layout()

fig1.savefig('figures/fig_vAbs_vRel_vEnv_compare_sim_02.pdf')

# ------------------ figure of v's dependence on T ---------------------------

fig2,ax2 = plt.subplots(1,1,figsize=[7,7])
ax2.scatter(abs_fit_clss,coeff_dT_a,color="blue",linewidth=2,label=r'$dv_a/dT$')
ax2.scatter(abs_fit_clss,coeff_dT_r,color="red",linewidth=2,label=r'$dv_r/dT$')
ax2.set_ylim(0,0.6*np.max(coeff_dT_a+coeff_dT_r))
ax2.set_xlabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax2.set_ylabel(r'Coefficient of $dv/dT$',fontsize=18,labelpad=8)
ax2.ticklabel_format(style='sci',axis='y',scilimits=None)
ax2.legend()
plt.text(-35,0.07e-5 ,r'$i_{max} = -98$',fontsize=16)
plt.text(-35,0.04e-5 ,r'$i_{ext} \approx -64$',fontsize=16)
plt.tight_layout()

fig2.savefig('figures/fig_vAbs_vRel_derivative_sim_02.pdf')

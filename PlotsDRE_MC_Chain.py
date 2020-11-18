# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:32:07 2020

@author: wyseq
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:23:46 2020

@author: wyseq
"""


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
# my functions in a seperate file
import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy.optimize import root 
import fig_functions_DRE as myfun            # my functions in a seperate file

# reload(myfun)
# parameters
T = 1e9         # carrying capacity
b=2.0           # juvenile birth rate
do=100/98.0      # minimum death rate / death rate of optimal genotype
R=1/130.0       # rate of environmental change

sa = 1e-2       # selection coefficient of beneficial mutation in 
Ua = 2e-5       # beneficial mutation rate in trait "d"   # It seems the mutation rate changes the direction of the intersections
Uad = 2e-5      # deleterious mutation rate in trait "d"

Ur=1e-5         # beneficial mutation rate in trait "c"
Urd=1e-5        # deleterious mutation rate in trait "c"
sr = 0.175     # multiplicative increment to "c" is (1+sr)

yi_option = 3   # select option to get equilibrium population density (=1,2,3)

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

max_index = 400
de = b+1
geompar = myfun.get_geom_death_pars(sa,de,do)
alpha = geompar[0]
geo = geompar[1]

di_fit_clss = []
for i in range(1,max_index):
    di_fit_clss.append(myfun.get_class_death_rate_DRE(alpha, geo, de, i))

def V_a_r_Extract(t, max_index):
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
    
    for i in range(1,max_index):
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = Ua # (400 - i) * Ua #i*Ua
        #sai = myfun.get_a_selection_coefficient_DRE(alpha,R,i)
        eqyi = myfun.get_eq_pop_density(b,do,sa,i,alpha,geo,de,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = myfun.get_a_selection_coefficient_DRE(alpha,geo,i)
        
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
    #abs_fit_clss = np.asarray(abs_fit_clss)
    #Ua_i = np.asarray(Ua_i)
    #eq_yi = np.asarray(eq_yi)
    #eq_Ni = np.asarray(eq_Ni)
    eff_sr_i = np.asarray(eff_sr_i)
    eff_sa_i = np.asarray(eff_sa_i)
    va_i = np.asarray(va_i)
    vr_i = np.asarray(vr_i)
    
    return [va_i, vr_i, eff_sr_i, eff_sa_i]

def inter_vs(t):
    def vsfun(i):
        
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = Ua
        eqyi = myfun.get_eq_pop_density(b,do,sa,i,alpha,geo,de,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = myfun.get_a_selection_coefficient_DRE(alpha,geo,i)
        
        # rates of fitness change (relative scale)
        vai = myfun.get_vDF(eqNi,effsai,Uai)
        vri = myfun.get_vDF(eqNi,effsri,Ur)
        
        return (vai - vri)
    
    return root(vsfun,x0=-38).x[0]

def V_a_r_Coeff(t, i_ext):
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
    
    for i in range(-i_ext+1,0):
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,do,sa,inter,alpha,geo,de,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
        
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
    
    coeff_dT_a = (eff_sa_i/(np.log(eff_sa_i/Ua_i)))**2 
    coeff_dT_r = (eff_sr_i/(np.log(eff_sr_i/Ur)))**2 
    
    return [coeff_dT_a, coeff_dT_r]

def V_a_r_Diff(t, i_ext):
    v = []
    
    for i in range(-i_ext+1,0):
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,do,sa,inter,alpha,geo,de,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
            
        # rates of fitness change (relative scale)
        v = v + [myfun.get_vDF(eqNi,effsai,Uai) - myfun.get_vDF(eqNi,effsri,Ur)]
        
    
    return (v)

# Intersection plot for diminishing returns model
bot = 5
up = 10

inter_v = []
inter_i = []
inter_di = []

fig1,ax1 = plt.subplots(1,1,figsize=[7,7])
colors = ['green', 'orange', 'brown', 'black','tan']
for t in range(bot,up):
    
    Vs = V_a_r_Extract(t, max_index)
    Va = Vs[0]
    Vr = Vs[1]

    inter = int(round(inter_vs(t)))
    
    inter_yi = myfun.get_eq_pop_density(b,do,sa,inter,alpha,geo,de,yi_option)
    inter_logNi = t*np.log(10) + np.log(inter_yi)
    inter_Ua = Ua#-inter*Ua
    
    inter_i.append(inter)
    inter_v.append(myfun.get_vDF(inter_logNi, myfun.get_a_selection_coefficient_DRE(alpha,geo,inter), inter_Ua))  
    inter_di.append(myfun.get_class_death_rate_DRE(alpha, geo, de, inter))
    #ax1.scatter(range(-i_ext+1,0),Va,color="blue",linewidth=2)
    ax1.plot(di_fit_clss,Va, color = colors[t-bot],linewidth=2, label = 'T = 10**' + str(t))
    
    #ax1.scatter(range(-i_ext+1,0),Vr,color="red",linewidth=2)
    ax1.plot(di_fit_clss,Vr,color = colors[t-bot],linewidth=2)

ax1.scatter(inter_di,inter_v,color="purple",linewidth=2)
ax1.plot(inter_di,inter_v,color="purple",linewidth=2)

ax1.set_xlabel(r'Death Rates for Absolute fitness class',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Rate of Adaptation',fontsize=18,labelpad=8)
ax1.set_ylim(0,.00002)
#ax1.set_ylim(0,1.25*np.max(Vr))
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()

plt.tight_layout()  





# Asymptote in the difference

# Pivot postions plot
i_ext = myfun.get_extinction_class(b,d,sa)

fig1,ax1 = plt.subplots(1,1,figsize=[7,7])

v = V_a_r_Diff(5,i_ext)
ax1.scatter(range(-i_ext+1,0),v,color="blue",linewidth=2, label = 'T = 10**5')
ax1.plot(range(-i_ext+1,0),v,vaRat,color="blue",linewidth=2)

v = V_a_r_Diff(9,i_ext)
ax1.scatter(range(-i_ext+1,0),v,color="red",linewidth=2, label = 'T = 10**9')
ax1.plot(range(-i_ext+1,0),v,vaRat,color="red",linewidth=2)

v = V_a_r_Diff(14,i_ext)
ax1.scatter(range(-i_ext+1,0),v,color="green",linewidth=2, label = 'T = 10**14')
ax1.plot(range(-i_ext+1,0),v,vaRat,color="green",linewidth=2)


ax1.plot(range(-i_ext+1,0),[0]*len(range(-i_ext+1,0)),color="black",linewidth=2)
ax1.set_xlabel(r'Absolute fitness Class',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Difference between Va and Vr',fontsize=18,labelpad=8)
ax1.set_ylim(np.min(v),np.max(v))
#ax1.set_ylim(-0.0000075,0.0000075)
ax1.set_xlim(-i_ext+1,0)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()

plt.tight_layout()












# Asymptote in ratio


def V_a_r_Ratio(t, i_ext):
    v = []
    
    for i in range(-i_ext+1,0):
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,do,sa,inter,alpha,geo,de,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
        
        inter_yi = myfun.get_eq_pop_density(b,do,sa,inter,alpha,geo,de,yi_option)
        
        ##
        inter_logNi = t*np.log(10) + np.log(inter_yi)
        inter_Ua = Ua#-inter*Ua
        
        inter_i.append(inter)
        inter_v.append(myfun.get_vDF(inter_logNi, myfun.get_a_selection_coefficient_DRE(alpha,geo,inter), inter_Ua))  
        inter_di.append(myfun.get_class_death_rate_DRE(alpha, geo, de, inter))
        ##
        
        # rates of fitness change (relative scale)
        v = v + [myfun.get_vDF(eqNi,effsai,Uai) / myfun.get_vDF(eqNi,effsri,Ur)]
        
    
    return (v)

i_ext = myfun.get_extinction_class(b,d,sa)

fig1,ax1 = plt.subplots(1,1,figsize=[7,7])

v = V_a_r_Ratio(5,i_ext)
ax1.scatter(range(-i_ext+1,0),v,color="blue",linewidth=2, label = 'T = 10**5')
ax1.plot(range(-i_ext+1,0),v,color="blue",linewidth=2)

v = V_a_r_Ratio(9,i_ext)
ax1.scatter(range(-i_ext+1,0),v,color="red",linewidth=2, label = 'T = 10**9')
ax1.plot(range(-i_ext+1,0),v,color="red",linewidth=2)

v = V_a_r_Ratio(14,i_ext)
ax1.scatter(range(-i_ext+1,0),v,color="green",linewidth=2, label = 'T = 10**14')
ax1.plot(range(-i_ext+1,0),v,color="green",linewidth=2)


ax1.plot(range(-i_ext+1,0),[1]*len(range(-i_ext+1,0)),color="black",linewidth=2)
ax1.set_xlabel(r'Absolute fitness Class',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Ratio between Va and Vr',fontsize=18,labelpad=8)
ax1.set_ylim(np.min(v),np.max(v))
#ax1.set_ylim(0.8,1.4)
#ax1.set_xlim(-55,0)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()

plt.tight_layout()
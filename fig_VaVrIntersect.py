# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:15:00 2020

@author: dirge
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:59:26 2020

@author: Owner
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import fig_functions_NR as myfun            # my functions in a seperate file
import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy.optimize import root 

# FUNCTIONS 
# ---------------------------------------------------------------------------
def V_a_r_Extract(t, i_ext, b, d, sa, Ua, sr, Ur, R):
# Function that pulls the relative and absolute fitness velocities
#inputs
#t - log territory size
#i_ext - the extinction class for given b, d, T
#outputs
#va_i - abs. rates of adaptation for set of valid fitness classes
#vr_i - rel. rates of adaptation for set of valid fitness classes
    
    abs_fit_clss = []   # absolute fitness classes
    Ua_i = []       # absolute fitness mutation rate
    d_i = []        # death rates for each of the absolute fitness classes
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
        eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
        d_i = d_i +[-d*(1+sa)**(-i) ]
        
        # rates of fitness change (relative scale)
        vai = myfun.get_vDF_lnN_vers(eqNi,effsai,Uai)
        vri = myfun.get_vDF_lnN_vers(eqNi,effsri,Ur)
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
        
    va_i = np.asarray(va_i)
    vr_i = np.asarray(vr_i)
    
    return [va_i, vr_i, d_i]

# ---------------------------------------------------------------------------
def inter_vs(t,b,d,sa,Ua,sr,Ur,fitclass):
# Extracts the intersection point between the relative and absolute fitness velocities
#inputs
#t-log10 territory size
#output
#i* - equilibrium fitness class
    
    def vsfun(i):
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
        
        # rates of fitness change (relative scale)
        vai = myfun.get_vDF_lnN_vers(eqNi,effsai,Uai)
        vri = myfun.get_vDF_lnN_vers(eqNi,effsri,Ur)
        
        return (vai - vri)
    
    return root(vsfun,x0=-fitclass).x[0]

# ---------------------------------------------------------------------------

def V_a_r_Diff(t, i_ext,b,d,sa,Ua,sr,Ur):
# function to get 
#inputs
#t - log10 territory size
#i_ext - extinction class
#outputs    
#v - difference between abs.rate of adapt. and rel.rate of adapt
    v = []
    for i in range(-i_ext+1,0):
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
            
        # rates of fitness change (relative scale)
        v = v + [myfun.get_vDF_lnN_vers(eqNi,effsai,Uai) - myfun.get_vDF_lnN_vers(eqNi,effsri,Ur)]
        
    return (v)

# ---------------------------------------------------------------------------
def get_tau_ratio(sa,Ua,sr,Ur):
#calculate ratio of tau's
#inputs
#s,U - selection coeff & mutation rates a-abs fit, r-rel fit
#outputs
# ratio of taus, as defined in Nathan et al. 2021
    tau_a = (1/sa)*np.log(sa/Ua)
    tau_r = (1/sr)*np.log(sr/Ur)
    return tau_a/tau_r

# ---------------------------------------------------------------------------

# **********************************************************************
#    Panel Plot (2 figures)
# **********************************************************************
fig1,ax1 = plt.subplots(1,1,figsize=[7,7])

# VARIABLES
T = 1e9         # carrying capacity
b=2.0           # juvenile birth rate
d=100/98.0      # minimum death rate / death rate of optimal genotype
R=1/130.0       # rate of environmental change

sa = 1e-2       # selection coefficient of beneficial mutation in 
Ua = 1e-6    # beneficial mutation rate in trait "d"
Uad = 1e-6      # deleterious mutation rate in trait "d"

Ur=1e-5         # beneficial mutation rate in trait "c"
Urd=1e-5        # deleterious mutation rate in trait "c"
sr = 0.175     # multiplicative increment to "c" is (1+sr)

yi_option = 3	# Numeric solution

# Graph for changing intersection under mutation model
bot = 5
up = 8

v_colors = ['green', 'orange', 'teal']
inter_color = ['black']

i_ext = myfun.get_extinction_class(b,d,sa)  

inter_v = []
inter_i = []
inter_di = []

interPts_v = []
interPts_i = []
interPts_di = []
interPts_ratio = []

# add intersection of va and vr curves 
for t in np.linspace(4.8,12,(11-5)/0.2):
    # get intersection points and append to list
    inter = inter_vs(t,b,d,sa,Ua,sr,Ur,38)
    inter_d = -d*(1+sa)**(-inter)
    inter_yi = myfun.get_eq_pop_density(b,d,sa,-inter,yi_option)[0]
    inter_logNi = t*np.log(10) + np.log(inter_yi)
    inter_Ua = -inter*Ua
    inter_i.append(inter)
    inter_di.append(inter_d)
    inter_v.append(myfun.get_vDF_lnN_vers(inter_logNi, sa, inter_Ua)  * 10**4)

# add va and vr curves 
for k,t in enumerate([5,7,9]):
    # get va,vr, and di's of Markov chain approx
    Vs = V_a_r_Extract(t, i_ext,b,d,sa,Ua,sr,Ur,R)
    Va = Vs[0] * 10**4
    Vr = Vs[1] * 10**4    
    d_i = Vs[2]
    
    interPts = inter_vs(t,b,d,sa,Ua,sr,Ur,38)
    interPts_d = -d*(1+sa)**(-interPts)
    interPts_yi = myfun.get_eq_pop_density(b,d,sa,-interPts,yi_option)[0]
    interPts_logNi = t*np.log(10) + np.log(interPts_yi)
    interPts_Ua = -interPts*Ua
    interPts_effsr = myfun.get_c_selection_coefficient(b,interPts_yi,sr)
    
    interPts_i.append(interPts)
    interPts_v.append(myfun.get_vDF_lnN_vers(interPts_logNi, sa, interPts_Ua)  * 10**4)  
    interPts_di.append(interPts_d)
    interPts_ratio.append(get_tau_ratio(sa,interPts_Ua,interPts_effsr,Ur))
    
    # add plots of va and vr
    current_label = r'$T=10^{}$'.format(t)
    ax1.plot(d_i,Va, color = v_colors[k],linewidth=2, label = current_label)    
    ax1.plot(d_i,Vr,color = v_colors[k],linewidth=2,linestyle = 'dashed')

asym_d = np.asarray([interPts_di[-1]-0.03 for i in range(0,17)])
asym_inter_v = np.linspace(0,1.4,17)

ax1.scatter(interPts_di,interPts_v,color="Black",linewidth=4)
ax1.plot(inter_di,inter_v,color="Black",linewidth=2)
ax1.plot(asym_d,asym_inter_v,color="Black",linestyle="dotted",linewidth=2)
ax1.set_xlabel(r'Death term (d)',fontsize=22,labelpad=8)
ax1.set_ylabel(r'Rate of Adaptation',fontsize=22,labelpad=8)
ax1.set_xlim(-3.1,-0.8)
ax1.set_ylim(0,np.max(1.25*Va))
ax1.set_xticks([-3.0+.5*i for i in range(0,5)])
ax1.set_xticklabels([3.0-.5*i for i in range(0,5)])
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.tick_params(labelsize=18)
ax1.legend(fontsize=18)
plt.text(-3.35,1.32,r'$\times 10^{-4}$', fontsize = 20)

myLabel1 = r'$\tau_a/\tau_r={}$'.format(np.round(interPts_ratio[0],2))
myLabel2 = r'$\tau_a/\tau_r={}$'.format(np.round(interPts_ratio[1],2))
myLabel3 = r'$\tau_a/\tau_r={}$'.format(np.round(interPts_ratio[2],2))

plt.text(interPts_di[0]-1.3,interPts_v[0]+.06,myLabel1, fontsize = 20, color = v_colors[0])
plt.text(interPts_di[1]-1.2,interPts_v[1]+.13,myLabel2, fontsize = 20, color = v_colors[1])
plt.text(interPts_di[2]-1.2,interPts_v[2]+.07,myLabel3, fontsize = 20, color = v_colors[2])

# add extra legend to point out that solid line is for absolute fitness 
# rate of adaptation and dashed is for relative fitness rate of adaptation
ax2 = ax1.twinx()
ax2.set_xlim(-3.1,-0.8)
ax2.set_ylim(0,np.max(1.25*Va))
ax2.set_yticklabels([])

plot_lines = []
plt.hold(True)
c = v_colors[0]
l1, = plt.plot([0],[1], '-', color=c)
l2, = plt.plot([0],[2], '--', color=c)

plot_lines.append([l1, l2,])

legend1 = plt.legend(plot_lines[0], ["Abs.Fit.", "Rel.Fit"], loc=2,fontsize=16)
#plt.legend([l[0] for l in plot_lines], [1 2], loc=4)
plt.gca().add_artist(legend1)

plt.tight_layout()
fig1.savefig('fig_decrAbsAdaptTerritorySize.pdf')

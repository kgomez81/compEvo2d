# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:02:44 2020

@author: wyseq
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import fig_functions as myfun            # my functions in a seperate file
import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy.optimize import root
# reload(myfun)
# parameters
T = 1e9         # carrying capacity
b=2.0          # juvenile birth rate
d=100/98.0      # minimum death rate / death rate of optimal genotype
R=1/130.0       # rate of environmental change

sa = 1e-2       # selection coefficient of beneficial mutation in 
Ua = 1e-6       # beneficial mutation rate in trait "d"
Uad = 1e-5      # deleterious mutation rate in trait "d"

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

inter_i = []


prob_incr_abs = []    # transition probability i->i-1
prob_decr_abs = []    # transition probability i->i+1

i_ext = myfun.get_extinction_class(b,d,sa)

inter_i = []
inter_ii = []
inter_i.append(-38.0)
inter_ii.append(-38.0)
for t in range(1,21):
    def vsfun(i):
    
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
        eqNi = 10**t*eqyi
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
        
        # rates of fitness change (relative scale)
        vai = myfun.get_vDF(eqNi,effsai,Uai)
        vri = myfun.get_vDF(eqNi,effsri,Ur)
        
        return (vai - vri)[0]
    inter_i.append(1.0*round(fsolve(vsfun,inter_i[t-1])[0]))
    inter_ii.append(1.0*round(root(vsfun,x0=inter_i[t-1])))
    
def vsfun(i):
    
    # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
    Uai = -i*Ua
    eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
    eqNi = 1000000000000*np.log(10) + np.log(eqyi)
    effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
    effsai = sa
    
    # rates of fitness change (relative scale)
    vai = myfun.get_vDF(eqNi,effsai,Uai)
    vri = myfun.get_vDF(eqNi,effsri,Ur)
    
    return (vai - vri)
kk = root(vsfun,x0=-40.0).x[0]
fsolve(vsfun,-38)
    
    
inter_ii = []
inter_i = []
for t in range(5,20):
    print(t)
    def vsfun(i):
        
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b,d,sa,-i,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
        effsai = sa
        
        # rates of fitness change (relative scale)
        vai = myfun.get_vDF(eqNi,effsai,Uai)
        vri = myfun.get_vDF(eqNi,effsri,Ur)
        
        return (vai - vri)
    
    inter_i.append(root(vsfun,x0=-38).x[0])
    
inter_ii = []
for t in range(5,20):
    print(t)
    def vsfun(i):
        
        # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
        Uai = -i*Ua
        eqyi = myfun.get_eq_pop_density(b+0.1,d,sa,-i,yi_option)
        eqNi = t*np.log(10) + np.log(eqyi)
        effsri = myfun.get_c_selection_coefficient(b+0.1,eqyi,sr)
        effsai = sa
        
        # rates of fitness change (relative scale)
        vai = myfun.get_vDF(eqNi,effsai,Uai)
        vri = myfun.get_vDF(eqNi,effsri,Ur)
        
        return (vai - vri)
    
    inter_ii.append(root(vsfun,x0=-38).x[0])
# ------------------ figure of Variation in T --------------------
fig1,ax1 = plt.subplots(1,1,figsize=[7,7])
ax1.scatter(range(5,20),inter_i,color="black",linewidth=2)
ax1.plot(range(5,20),inter_i,color="black",linewidth=2)
ax1.scatter([19],[np.min(inter_i)],color="red",linewidth=2)
ax1.scatter([9],[-38.0429357890911],color="blue",linewidth=2)

#ax1.set_ylim(1.25*np.min(inter_ii),1.25*np.max(inter_i))
ax1.set_xlabel(r'Log10(Territory Size)',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
plt.text(15 ,-38.4,r'$i_{int} \approx -38.707$',fontsize=16)
plt.text(9 ,-37.92,r'$i_{int} \approx -38.042$',fontsize=16)

plt.tight_layout()
fig1.savefig('figures/V_Intercept_TerritorySize.jpg')
fig1.savefig('figures/V_Intercept_TerritorySize.pdf')
fig1.savefig('figures/V_Intercept_TerritorySize.jpg')
# ------------------ figure of Variation in Birthrate --------------------
def vscurveb(B, up):
    inter_i = []
    for t in range(5,up):
        def vsfun(i):
            
            # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
            Uai = -i*Ua
            eqyi = myfun.get_eq_pop_density(B,d,sa,-i,yi_option)
            eqNi = t*np.log(10) + np.log(eqyi)
            effsri = myfun.get_c_selection_coefficient(B,eqyi,sr)
            effsai = sa
            
            # rates of fitness change (relative scale)
            vai = myfun.get_vDF(eqNi,effsai,Uai)
            vri = myfun.get_vDF(eqNi,effsri,Ur)
            
            return (vai - vri)
        
        inter_i.append(root(vsfun,x0=-38).x[0])
    return inter_i

greys = ['whitesmoke','gainsboro','lightgrey','lightgray', 'silver','darkgrey','darkgray','grey','gray','dimgrey','dimgray','black']

# ------------------ figure of Variation in Birthrate Plot 1 --------------------
fig1,ax1 = plt.subplots(1,1,figsize=[8,8])
ax1.scatter(range(5,20),inter_i,color="blue",linewidth=2, label = 'b = 2.0 (orig)')
ax1.plot(range(5,20),inter_i,color="blue",linewidth=2)
for i in range(1,12):
    ax1.scatter(range(5,20),vscurveb(2.0 - i * 0.02, 20),color=greys[-i],linewidth=2, label='b =' + str(2.0 - i * 0.02))
    ax1.plot(range(5,20),vscurveb(2.0 - i * 0.02, 20),color=greys[-i],linewidth=2)


#ax1.plot([np.min(inter_i), -38.0429357890911],[9, 9],color="purple",linewidth=2)
#ax1.plot([np.min(inter_i), np.min(inter_i)],[9, 19],color="purple",linewidth=2)
#ax1.set_ylim(1.25*np.min(inter_ii),1.25*np.max(inter_i))
ax1.set_xlabel(r'Log10(Territory Size)',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
#plt.text(-38.6,19 ,r'$i_{int} \approx -38.707$',fontsize=16)
#plt.text(-37.92,9 ,r'$i_{int} \approx -38.042$',fontsize=16)

plt.tight_layout()
fig1.savefig('figures/V_Intercept_Birth_VariationR.jpg')
fig1.savefig('figures/V_Intercept_Birth_VariationR.pdf')


# ------------------ figure of Variation in Birthrate Plot 2 --------------------
greys = ['whitesmoke','gainsboro','lightgrey','lightgray', 'silver','darkgrey','darkgray','grey','gray','dimgrey','dimgray','black']


fig1,ax1 = plt.subplots(1,1,figsize=[8,8])
ax1.scatter(range(5,20),inter_i,color="blue",linewidth=2, label = 'b = 2.0 (orig)')
ax1.plot(range(5,20),inter_i,color="blue",linewidth=2)
for i in range(1,12):
    ax1.scatter(range(5,20),vscurveb(2.0 + i * 0.02, 20),color=greys[-i],linewidth=2, label='b =' + str(2.0 + i * 0.02))
    ax1.plot(range(5,20),vscurveb(2.0 + i * 0.02, 20),color=greys[-i],linewidth=2)

#ax1.plot([np.min(inter_i), -38.0429357890911],[9, 9],color="purple",linewidth=2)
#ax1.plot([np.min(inter_i), np.min(inter_i)],[9, 19],color="purple",linewidth=2)
#ax1.set_ylim(1.25*np.min(inter_ii),1.25*np.max(inter_i))
ax1.set_xlabel(r'Log10(Territory Size)',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
#plt.text(-38.6,19 ,r'$i_{int} \approx -38.707$',fontsize=16)
#plt.text(-37.92,9 ,r'$i_{int} \approx -38.042$',fontsize=16)

plt.tight_layout()
fig1.savefig('figures/V_Intercept_Birth_VariationL.jpg')
fig1.savefig('figures/V_Intercept_Birth_VariationL.pdf')


# ------------------ figure of Variation in Birthrate --------------------
def vscurved(D, up):
    inter_i = []
    for t in range(5,up):
        def vsfun(i):
            
            # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr
            Uai = -i*Ua
            eqyi = myfun.get_eq_pop_density(b,D,sa,-i,yi_option)
            eqNi = t*np.log(10) + np.log(eqyi)
            effsri = myfun.get_c_selection_coefficient(b,eqyi,sr)
            effsai = sa
            
            # rates of fitness change (relative scale)
            vai = myfun.get_vDF(eqNi,effsai,Uai)
            vri = myfun.get_vDF(eqNi,effsri,Ur)
            
            return (vai - vri)
        
        inter_i.append(root(vsfun,x0=-38).x[0])
    return inter_i

greys = ['whitesmoke','gainsboro','lightgrey','lightgray', 'silver','darkgrey','darkgray','grey','gray','dimgrey','dimgray','black']

# ------------------ figure of Variation in Deathrate Plot 1 --------------------
fig1,ax1 = plt.subplots(1,1,figsize=[8,8])
ax1.scatter(range(5,20),inter_i,color="blue",linewidth=2, label = 'd = 100/98 (orig)')
ax1.plot(range(5,20),inter_i,color="blue",linewidth=2)
for i in range(1,7):
    ax1.scatter(range(5,20),vscurved(100/(98.0+i), 20),color=greys[-i],linewidth=2, label='d = 100/' + str((98 + i)))
    ax1.plot(range(5,20),vscurved(100/(98.0+i), 20),color=greys[-i],linewidth=2)


#ax1.plot([np.min(inter_i), -38.0429357890911],[9, 9],color="purple",linewidth=2)
#ax1.plot([np.min(inter_i), np.min(inter_i)],[9, 19],color="purple",linewidth=2)
#ax1.set_ylim(1.25*np.min(inter_ii),1.25*np.max(inter_i))
ax1.set_xlabel(r'Log10(Territory Size)',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
#plt.text(-38.6,19 ,r'$i_{int} \approx -38.707$',fontsize=16)
#plt.text(-37.92,9 ,r'$i_{int} \approx -38.042$',fontsize=16)

plt.tight_layout()
fig1.savefig('figures/V_Intercept_Death_VariationL.jpg')
fig1.savefig('figures/V_Intercept_Death_VariationL.pdf')


# ------------------ figure of Variation in Deathrate Plot 2 --------------------
fig1,ax1 = plt.subplots(1,1,figsize=[8,8])
ax1.scatter(range(5,20),inter_i,color="blue",linewidth=2, label = 'd = 100/98 (orig)')
ax1.plot(range(5,20),inter_i,color="blue",linewidth=2)
for i in range(1,12):
    ax1.scatter(range(5,20),vscurved(100/(98.0-i), 20),color=greys[-i],linewidth=2, label='d = 100/' + str((98 - i)))
    ax1.plot(range(5,20),vscurved(100/(98.0-i), 20),color=greys[-i],linewidth=2)


#ax1.plot([np.min(inter_i), -38.0429357890911],[9, 9],color="purple",linewidth=2)
#ax1.plot([np.min(inter_i), np.min(inter_i)],[9, 19],color="purple",linewidth=2)
#ax1.set_ylim(1.25*np.min(inter_ii),1.25*np.max(inter_i))
ax1.set_xlabel(r'Log10(Territory Size)',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
#plt.text(-38.6,19 ,r'$i_{int} \approx -38.707$',fontsize=16)
#plt.text(-37.92,9 ,r'$i_{int} \approx -38.042$',fontsize=16)

plt.tight_layout()
fig1.savefig('figures/V_Intercept_Death_VariationR.jpg')
fig1.savefig('figures/V_Intercept_Death_VariationR.pdf')


# ------------------ test area --------------------
fig1,ax1 = plt.subplots(1,1,figsize=[8,8])
ax1.scatter(range(5,20),inter_i,color="blue",linewidth=2, label = 'd = 100/98 (orig)')
ax1.plot(range(5,20),inter_i,color="blue",linewidth=2)
for i in range(1,12):
    ax1.scatter(range(5,20),vscurved(100/(98.0-i), 20),color=greys[-i],linewidth=2, label='d = 100/' + str((98 - i)))
    ax1.plot(range(5,20),vscurved(100/(98.0-i), 20),color=greys[-i],linewidth=2)


#ax1.plot([np.min(inter_i), -38.0429357890911],[9, 9],color="purple",linewidth=2)
#ax1.plot([np.min(inter_i), np.min(inter_i)],[9, 19],color="purple",linewidth=2)
#ax1.set_ylim(1.25*np.min(inter_ii),1.25*np.max(inter_i))
ax1.set_xlabel(r'Log10(Territory Size)',fontsize=18,labelpad=8)
ax1.set_ylabel(r'Absolute fitness class',fontsize=18,labelpad=8)
ax1.ticklabel_format(style='sci',axis='y',scilimits=None)
ax1.legend()
#plt.text(-38.6,19 ,r'$i_{int} \approx -38.707$',fontsize=16)
#plt.text(-37.92,9 ,r'$i_{int} \approx -38.042$',fontsize=16)


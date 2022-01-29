# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import fig_functions_NR as myfun            # my functions in a seperate file
import csv

# load parameters from evoParameters.csv file. The parameters must be provided in the following 
# order (numeric values). All input csv files must be available in working directory containing scripts.
# Below are examples of standard parameters initially used in simulations
#T   = 1e9           # carrying capacity
#b   = 2.0            # birth rate
#do  = 100/98.0   # minimum death rate / death rate of optimal genotype
#sa  = 1e-2          # selection coefficient of beneficial mutation in 
#Ua  = 1e-5         # max beneficial mutation rate in trait "d"
#Uad = 1e-5       # deleterious mutation rate in trait "d"
#cr  = 0.175       # increment to "c" is (1+cr)
#Ur  = 1e-5         # beneficial mutation rate in trait "c"
#Urd = 1e-5       # deleterious mutation rate in trait "c"
#R1  = 1/130.0   # rate of environmental change per iteration
#R2  = 1/125.0   # rate of environmental change per iteration

paramFile = 'evoParameters.csv'
paramList = ['T','b','dOpt','sa','UaMax','Uad','cr','Ur','Urd','R1','R2']

# function read_param_file returns a dictionary listing all parameters in
# paramList and with values read from the paramFile.
params = myfun.read_param_file(paramFile,paramList)

# get the parameters for the absolute fitness state space
[iMax,dMax,di] = myfun.get_extinction_classes( )

# import pfix values from simulations. if 0,...,iMax are the number of absolute
# fitness states, then pFix values must be specified for 0,...,(iMax-1), i.e.
# i.e. each possible beneficial mutation.
pFix_Abs_File = 'parallel_abs.csv'
pFix_Abs_Values = myfun.read_pFixOutputs(pFix_Abs_File,iMax-1)

# import pfix values from simulations. if 0,...,iMax are the number of absolute
# fitness states, then pFix values must be specified for 1,...,iMax states
# i.e. each possible relative fitness beneficial mutation competing against an
# absolute fitness beneficial mutation.
pFix_Abs_File = 'parallel_rel.csv'
pFix_Abs_Values = myfun.read_pFixOutputs(pFix_Abs_File,iMax-1)

# set root solving option for equilibrium densities
yi_option = 3   # select option to get equilibrium population density (=1,2,3)

# Calculate all evolution parameters.
iState  = []    # state number
Ua_i    = []    # absolute fitness mutation rate
Ur_i    = []    # relative fitness mutation rate
eq_yi   = []    # equilibrium density of fitness class i
eq_Ni   = []    # equilibrium population size of fitness class i
sr_i    = []    # selection coefficient of "c" trait beneficial mutation
sa_i    = []    # selection coefficient of "c" trait beneficial mutation

va_i    = []    # rate of adaptation in absolute fitness trait alone
vr_i    = []    # rate of adaptation in relative fitness trait alone
ve_i    = []    # rate of fitness decrease due to environmental degradation

prob_incr_abs = []    # transition probability i->i-1
prob_decr_abs = []    # transition probability i->i+1

# calculate evolution parameter for each of the states in the markov chain model
# the evolution parameters are calculated along the absolute fitness state space
# beginning with state 1 (1 mutation behind optimal) to iExt (extinction state)
for i in range(1,iExt+1):
    # absolute fitness mutation rate, equilb.-density,equilb.-popsize,eff_sr 
    iState  = -i
    Ua_i    = Ua_i + [params['UaMax']*(i/iExt)]
    Ur_i    = Ur_i + [params['Ur']]
    eq_yi   = eq_yi + [myfun.get_eq_pop_density(params['b'],params['dOpt'],params['sa'],i,yi_option)]
    eq_Ni   = eq_Ni + [params['T']*eq_yi[-1]]
    sr_i    = sr_i + [myfun.get_c_selection_coefficient(params['b'],eq_yi[-1],params['cr'],di[i-1])]
    sa_i    = params['sa']
    
    # rates of fitness change ( on time scale of generations)
    va_i = va_i + [myfun.get_rateOfAdapt(eq_Ni[-1],sa_i[-1],Ua_i[-1])]
    vr_i = vr_i + [myfun.get_rateOfAdapt(eq_Ni[-1],sr_i[-1],Ur[-1]  )]
    ve_i = ve_i + [sa_i[-1]*R/(di[-1]-1)]  
    
# state space
iState  = np.asarray(iState)
Ua_i    = np.asarray(Ua_i)
Ur_i    = np.asarray(Ur_i)
eq_yi   = np.asarray(eq_yi)
eq_Ni   = np.asarray(eq_Ni)
sr_i    = np.asarray(eff_sr_i)
sa_i    = np.asarray(eff_sa_i)

va_i    = np.asarray(va_i)
vr_i    = np.asarray(vr_i)
ve_i    = np.asarray(ve_i)

# ------------------ figure of Markov chain approximation --------------------
#
# Figure for scenario 1 - Intersection of va and ve
#
fig1,ax1 = plt.subplots(1,1,figsize=[7,7])
ax1.scatter(d_i,va_i,color="blue",linewidth=1.0,label=r'$v_a$')
ax1.scatter(d_i,vr_i,color="red",linewidth=1.0,label=r'$v_r$')
ax1.plot(d_i,ve_i,color="green",linewidth=2,label=r' $R_{1}$')
ax1.plot(d_i,(130.0/103.0)*ve_i,color="black",linewidth=2,label=r'$R_{2}$')
ax1.scatter([-1.89],[sa/103.0],color="none",s=100,edgecolor="black",linewidth=2.0,marker='o')
ax1.scatter([-1.59],[sa/115],color="none",s=100,edgecolor="black",linewidth=2.0,marker='o')

#ax1.plot(d_i,ve_i,color="green",linewidth=2)
#ax1.plot(d_i,(130.0/103.0)*ve_i,color="black",linewidth=2)

ax1.set_xlim(-3.1,-0.8)
ax1.set_ylim(0,np.max(1.25*va_i))
ax1.set_xticks([-3.0+.5*i for i in range(0,5)])
ax1.set_xticklabels([3.0-.5*i for i in range(0,5)],fontsize=18)

ax1.set_yticks([10**(-4)*(0+0.2*i) for i in range(0,7)])
ax1.set_yticklabels([str(0+0.2*i) for i in range(0,7)],fontsize=18)

ax1.set_xlabel(r'Death term (d)',fontsize=22,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=22,labelpad=8)
#ax1.ticklabel_format(style='sci',axis='y',scilimits=None)

handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]

ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize = 20,ncol=2)
#plt.text(-20,1.2e-5 ,r'$i_{max} = -98$',fontsize=16)
#plt.text(-20,0.5e-5 ,r'$i_{ext} \approx -64$',fontsize=16)

#ax1.plot([-1.89,-1.89],[0,sa/104],c="black",linewidth=2,linestyle='--')
#ax1.annotate("", xy=(-1.92,sa/106), xytext=(-2.1, sa/106),arrowprops={'arrowstyle':'-|>','lw':3})
#ax1.annotate("", xy=(-1.86,sa/100), xytext=(-1.69, sa/100),arrowprops={'arrowstyle':'-|>','lw':3})
#
#ax1.plot([-1.59,-1.59],[0,sa/116],c="black",linewidth=2,linestyle='--')
#ax1.annotate("", xy=(-1.62,sa/115), xytext=(-1.79, sa/115),arrowprops={'arrowstyle':'-|>','lw':3})
#ax1.annotate("", xy=(-1.55,sa/115), xytext=(-1.37, sa/115),arrowprops={'arrowstyle':'-|>','lw':3})

ax1.plot([-1.89,-1.89],[sa/1000,sa/104],c="black",linewidth=2,linestyle='--')
ax1.plot([-1.59,-1.59],[sa/300,sa/116],c="black",linewidth=2,linestyle='--')

ax1.annotate("", xy=(-1.92,sa/1000), xytext=(-2.1, sa/1000),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax1.annotate("", xy=(-1.86,sa/1000), xytext=(-1.69, sa/1000),arrowprops={'arrowstyle':'-|>','lw':3})

ax1.annotate("", xy=(-1.62,sa/300), xytext=(-1.79, sa/300),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax1.annotate("", xy=(-1.55,sa/300), xytext=(-1.37, sa/300),arrowprops={'arrowstyle':'-|>','lw':3,'color':'red'})

plt.text(-3.35,1.32*10**(-4),r'$\times 10^{-4}$', fontsize = 20)

# add extra legend to point out that solid line is for absolute fitness 
# rate of adaptation and dashed is for relative fitness rate of adaptation
#ax2 = ax1.twinx()
#ax2.set_xlim(-3.1,-0.8)
#ax2.set_ylim(0,np.max(1.25*Va))
#ax2.set_yticklabels([])
#
#plot_lines = []
#plt.hold(True)
#c = v_colors[0]
#l1, = plt.plot([0],[1], '-', color="Black")
#l2, = plt.plot([0],[2], '-', color="Green")
#
#plot_lines.append([l1, l2,])
#
#legend1 = plt.legend(plot_lines[0], [r'$R_{130}$', r'$R_{103}$'], loc=2,fontsize=16)
##plt.legend([l[0] for l in plot_lines], [1 2], loc=4)
#plt.gca().add_artist(legend1)

plt.tight_layout()
fig1.savefig('fig_MC_chain_VaVeIntersection.pdf')


# ------------------ figure of Markov chain approximation --------------------
#
# Scenario 2 - Intersection of va and ve

fig2,ax2 = plt.subplots(1,1,figsize=[7,7])
ax2.scatter(d_i,va_i,color="blue",linewidth=1.0,label=r'$v_a$')
ax2.scatter(d_i,vr_i,color="red",linewidth=1.0,label=r'$v_r$')
ax2.plot(d_i,ve_i,color="green",linewidth=2,label=r' $R_{1}$')
ax2.plot(d_i,(130.0/103.0)*ve_i,color="black",linewidth=2,label=r'$R_{2}$')
ax2.scatter([-1.89],[sa/103.0],color="none",s=100,edgecolor="black",linewidth=2.0,marker='o')
ax2.scatter([-1.59],[sa/115],color="none",s=100,edgecolor="black",linewidth=2.0,marker='o')


#ax1.plot(d_i,ve_i,color="green",linewidth=2)
#ax1.plot(d_i,(130.0/103.0)*ve_i,color="black",linewidth=2)

ax2.set_xlim(-3.1,-0.8)
ax2.set_ylim(0,np.max(1.25*va_i))
ax2.set_xticks([-3.0+.5*i for i in range(0,5)])
ax2.set_xticklabels([3.0-.5*i for i in range(0,5)],fontsize=18)

ax2.set_yticks([10**(-4)*(0+0.2*i) for i in range(0,7)])
ax2.set_yticklabels([str(0+0.2*i) for i in range(0,7)],fontsize=18)

ax2.set_xlabel(r'Death term (d)',fontsize=22,labelpad=8)
ax2.set_ylabel(r'Rate of adaptation',fontsize=22,labelpad=8)
#ax1.ticklabel_format(style='sci',axis='y',scilimits=None)

handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]

ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize = 20,ncol=2)
#plt.text(-20,1.2e-5 ,r'$i_{max} = -98$',fontsize=16)
#plt.text(-20,0.5e-5 ,r'$i_{ext} \approx -64$',fontsize=16)

#ax1.plot([-1.89,-1.89],[0,sa/104],c="black",linewidth=2,linestyle='--')
#ax1.annotate("", xy=(-1.92,sa/106), xytext=(-2.1, sa/106),arrowprops={'arrowstyle':'-|>','lw':3})
#ax1.annotate("", xy=(-1.86,sa/100), xytext=(-1.69, sa/100),arrowprops={'arrowstyle':'-|>','lw':3})
#
#ax1.plot([-1.59,-1.59],[0,sa/116],c="black",linewidth=2,linestyle='--')
#ax1.annotate("", xy=(-1.62,sa/115), xytext=(-1.79, sa/115),arrowprops={'arrowstyle':'-|>','lw':3})
#ax1.annotate("", xy=(-1.55,sa/115), xytext=(-1.37, sa/115),arrowprops={'arrowstyle':'-|>','lw':3})

ax1.plot([-1.89,-1.89],[sa/1000,sa/104],c="black",linewidth=2,linestyle='--')
ax1.plot([-1.59,-1.59],[sa/300,sa/116],c="black",linewidth=2,linestyle='--')

ax1.annotate("", xy=(-1.92,sa/1000), xytext=(-2.1, sa/1000),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax1.annotate("", xy=(-1.86,sa/1000), xytext=(-1.69, sa/1000),arrowprops={'arrowstyle':'-|>','lw':3})

ax1.annotate("", xy=(-1.62,sa/300), xytext=(-1.79, sa/300),arrowprops={'arrowstyle':'-|>','lw':3,'color':'blue'})
ax1.annotate("", xy=(-1.55,sa/300), xytext=(-1.37, sa/300),arrowprops={'arrowstyle':'-|>','lw':3,'color':'red'})

plt.text(-3.35,1.32*10**(-4),r'$\times 10^{-4}$', fontsize = 20)

# add extra legend to point out that solid line is for absolute fitness 
# rate of adaptation and dashed is for relative fitness rate of adaptation
#ax2 = ax1.twinx()
#ax2.set_xlim(-3.1,-0.8)
#ax2.set_ylim(0,np.max(1.25*Va))
#ax2.set_yticklabels([])
#
#plot_lines = []
#plt.hold(True)
#c = v_colors[0]
#l1, = plt.plot([0],[1], '-', color="Black")
#l2, = plt.plot([0],[2], '-', color="Green")
#
#plot_lines.append([l1, l2,])
#
#legend1 = plt.legend(plot_lines[0], [r'$R_{130}$', r'$R_{103}$'], loc=2,fontsize=16)
##plt.legend([l[0] for l in plot_lines], [1 2], loc=4)
#plt.gca().add_artist(legend1)

plt.tight_layout()
fig1.savefig('fig_MC_chain_VaVrIntersection.pdf')
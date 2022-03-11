# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 03:47:26 2022

@author: dirge
"""

# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import evo_library as myfun            # my functions in a seperate file

# --------------------------------------------------------------------------
# get parameters and array of values
# --------------------------------------------------------------------------

# The parameter file is read and a dictionary with their values is generated.
paramFile = 'inputs/evoExp02_parameters_VaVrIntersect.csv'
params = myfun.read_parameterFile(paramFile)

# set root solving option for equilibrium densities
# (1) low-density analytic approximation 
# (2) high-density analytic approximation
# (3) root solving numerical approximation
yi_option = 3  

T_vals = np.logspace(5,13,num=9)
sa_vals = np.logspace(-3,-1,num=20)

T_ARRY, SA_ARRY = np.meshgrid(T_vals, sa_vals)
RHO_ARRY = np.zeros(T_ARRY.shape)
Y_ARRY = np.zeros(T_ARRY.shape)

paramsTemp = params

# --------------------------------------------------------------------------
# Calculated rho values for T vs 2nd parameter variable
# --------------------------------------------------------------------------

for ii in range(int(T_ARRY.shape[0])):
    for jj in range(int(T_ARRY.shape[1])):
        
        paramsTemp['T'] = T_ARRY[ii,jj]
        paramsTemp['sa'] = SA_ARRY[ii,jj]
        
        # Calculate absolute fitness state space. 
        [dMax,di,iExt] = myfun.get_absoluteFitnessClasses(paramsTemp['b'],paramsTemp['dOpt'],paramsTemp['sa'])
        
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i] = myfun.get_MChainPopParameters(paramsTemp,di,iExt,yi_option)        
        
        pFixAbs_i = np.reshape(np.asarray(sa_i),[len(sa_i),1])
        pFixRel_i = np.reshape(np.asarray(sr_i),[len(sr_i),1])
        
        # Use s values for pFix until we get sim pFix values can be obtained
        [state_i,Ua_i,Ur_i,eq_yi,eq_Ni,sr_i,sa_i,va_i,vr_i,ve_i] = \
                        myfun.get_MChainEvoParameters(paramsTemp,di,iExt,pFixAbs_i,pFixRel_i,yi_option)
 
        RHO_ARRY[ii,jj] = myfun.get_intersection_rho(va_i, vr_i, sa_i, Ua_i, Ur_i, sr_i)   
        Y_ARRY[ii,jj] = myfun.get_intersection_popDensity(va_i, vr_i, eq_yi)   


# --------------------------------------------------------------------------
# Contour plot of rho values
# --------------------------------------------------------------------------

fig, ax1 = plt.subplots(1,1,figsize=[7,6],subplot_kw={"projection": "3d"})

cp = ax1.contourf(np.log10(T_ARRY), (1/params['cr'])*SA_ARRY, Y_ARRY)
#cp = ax1.plot_surface(np.log10(T_ARRY), (1/params['cr'])*SA_ARRY, Y_ARRY)
fig.colorbar(cp) # Add a colorbar to a plot
ax1.set_title('Rho Contours Plot')
ax1.set_xlabel('log10 T')
ax1.set_ylabel('sa/cr')

#ax1.plot(state_i,ve_i,color="black",linewidth=3,label=r'$v_e$')
#ax1.scatter(state_i,va_i,color="blue",s=8,label=r'$v_a$')
#ax1.scatter(state_i,vr_i,color="red",s=8,label=r'$v_r$')

## axes and label adjustements
#ax1.set_xlim(-iExt-1,0)
#ax1.set_ylim(0,2.52e-4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])
#ax1.set_xticks([-25*i for i in range(0,iExt/25+1)])
#ax1.set_xticklabels([str(25*i) for i in range(0,iExt/25+1)],fontsize=16)
##ax1.set_xticklabels(["" for i in range(0,iExt/25+1)],fontsize=16)
#ax1.set_yticks([1e-5*5*i for i in range(0,6)])
##ax1.set_yticklabels(["" for i in range(0,6)],fontsize=16)
#ax1.set_yticklabels([str(5*i/10.0) for i in range(0,6)],fontsize=16)
##ax1.set_xlabel(r'Absolute fitness class',fontsize=20,labelpad=8)
#ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)
#ax1.legend(fontsize = 14,ncol=1,loc='lower right')
#
## annotations
#ax1.plot([-88,-88],[0,1.6e-4],c="black",linewidth=2,linestyle='--')
#ax1.annotate("", xy=(-89,0.7e-4), xytext=(-104,0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4,'color':'blue'})
#ax1.annotate("", xy=(-87,0.7e-4), xytext=(-72, 0.7e-4),arrowprops={'arrowstyle':'-|>','lw':4})
##plt.text(-84,3.29e-4,r'$i^*=88$',fontsize = 18)
##plt.text(-84,3.10e-4,r'$i_{ext}=180$',fontsize = 18)
##plt.text(-190,5.50e-4,r'$\times 10^{-4}$', fontsize = 20)
#plt.text(-175,5.15e-4,r'(A)', fontsize = 22)



plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 20:56:50 2022

@author: Owner
"""

plt.plot(sa_i,sr_i)
plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

plt.plot(Ua_i,Ur_i)
plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

plt.plot(pFixAbs_i,pFixRel_i)
plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

plt.plot(eq_yi,eq_Ni)
plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

plt.scatter(state_i,va_i)
plt.scatter(state_i,vr_i)
plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))
plt.ylim([0,1.5*max([max(va_i),max(vr_i)])])


plt.scatter(di[1:],va_i)
plt.scatter(di[1:],vr_i)
plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))
plt.ylim([0,1.5*max([max(va_i),max(vr_i)])])


# pfix versus sel coeff
fig,ax = plt.subplots(1,1,figsize=[7,7])
ax.plot(mcModel1.state_i,mcModel1.pFix_c_i,color='red')
ax.plot(mcModel1.state_i,mcModel1.sc_i,color='blue')

fig,ax = plt.subplots(1,1,figsize=[7,7])
ax.plot(mcModel1.state_i,mcModel1.pFix_d_i,color='red')
ax.plot(mcModel1.state_i,mcModel1.sd_i,color='blue')

# -----------------------------------------------------

import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.getcwd()+'/inputs/evoExp_RM_01_parameters.csv'
modelType = 'RM'

mcParams_rm = evoObj.evoOptions(paramFilePath, modelType)
mcModel_rm = mcRM.mcEvoModel_RM(mcParams_rm.params)

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.getcwd()+'/inputs/evoExp_DRE_01_parameters.csv'
modelType = 'DRE'

mcParams_dre = evoObj.evoOptions(paramFilePath, modelType)
mcModel_dre = mcDRE.mcEvoModel_DRE(mcParams_dre.params)

# unscaled rates

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])

ax1.scatter(-mcModel_rm.di , mcModel_rm.get_vd_i_perUnitTime()   ,c='blue',label='vd')
ax1.scatter(-mcModel_rm.di , mcModel_rm.get_vc_i_perUnitTime()   ,c='red',label='vc')
ax1.plot(   -mcModel_rm.di , mcModel_rm.get_ve_i_perUnitTime()   ,c='black',label='ve')

ax1.set_ylim(0,9e-4)

ax2.scatter(-mcModel_dre.di, mcModel_dre.get_vd_i_perUnitTime()  ,c='blue',label='vd')
ax2.scatter(-mcModel_dre.di, mcModel_dre.get_vc_i_perUnitTime()  ,c='red',label='vc')
ax2.plot(   -mcModel_dre.di, mcModel_dre.get_ve_i_perUnitTime()  ,c='black',label='ve')

ax2.set_ylim(0,9e-4)

plt.legend()

# evo regime

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])

ax1.scatter(-mcModel_rm.di , mcModel_rm.evoRegime_d_i   ,c='blue',label='vd',marker = 'o')
ax1.scatter(-mcModel_rm.di , mcModel_rm.evoRegime_c_i   ,c='red',label='vc' ,marker = '*')

ax2.scatter(-mcModel_dre.di, mcModel_dre.evoRegime_d_i  ,c='blue',label='vd',marker = 'o')
ax2.scatter(-mcModel_dre.di, mcModel_dre.evoRegime_c_i  ,c='red',label='vc' ,marker = '*')


plt.legend()

# -----------------------------------------------------
# all three at once

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[7,10])
ax1.plot(mcModel1.state_i,np.log10(mcModel1.sd_i),c='blue')
ax1.plot(mcModel1.state_i,np.log10(mcModel1.sc_i),c='red')
ax1.set_ylabel('s')

ax2.plot(mcModel1.state_i,np.log10(mcModel1.pFix_d_i),c='blue')
ax2.plot(mcModel1.state_i,np.log10(mcModel1.pFix_c_i),c='red')
ax2.set_ylabel('$\pi_{fix}$')

ax3.plot(mcModel1.state_i,np.log10(mcModel1.vd_i),c='blue')
ax3.plot(mcModel1.state_i,np.log10(mcModel1.vc_i),c='red')
ax3.set_ylabel('v')

# -----------------------------------------------------
# s and pfix

fig,ax1 = plt.subplots(1,1,figsize=[7,5])
ax1.plot(mcModel1.state_i,np.log10(mcModel1.sd_i),c='blue',label='sd')
ax1.plot(mcModel1.state_i,np.log10(mcModel1.sc_i),c='red',label='sc')

ax1.plot(mcModel1.state_i,np.log10(mcModel1.pFix_d_i),c='green',label='pFix d')
ax1.plot(mcModel1.state_i,np.log10(mcModel1.pFix_c_i),c='purple',label='pFix c')
ax1.set_ylabel(r'$s$ and $\pi_{fix}$')

plt.legend()

# -----------------------------------------------------
# more general plots of quantities

plt.plot(mcModel1.sd_i,mcModel1.sc_i)

plt.scatter(mcModel1.state_i[0:100],np.log10(mcModel1.sd_i[0:100]))

plt.plot(mcModel1.state_i[0:100],mcModel1.di[0:100]-1.02)

plt.plot(mcModel1.state_i[0:100],np.log10(mcModel1.eq_Ni[0:100]))

plt.plot(mcModel1.state_i[0:99],np.log10(np.abs(mcModel1.di[1:100]-mcModel1.di[0:99])/mcModel1.di[0:99]) )

plt.plot(mcModel2.sd_i,mcModel2.sc_i)

plt.plot(mcModel2.vd_i,mcModel2.vc_i)

plt.scatter(mcModel2.Ud_i,mcModel2.Uc_i)

# -----------------------------------------------

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
ax1.scatter(mcModel_rm.state_i,mcModel_rm.di)
ax2.scatter(mcModel_dre.state_i,mcModel_dre.di)

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
ax1.scatter(mcModel_rm.di,mcModel_rm.sd_i)
ax2.scatter(mcModel_dre.di,mcModel_dre.sd_i)

# -----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def dj_terms(do,de,Fjj):
    
    dj = de*(do/de)**Fjj
    
    return dj

def selCoeff_k(dj1,dj2):
    
    sj = (dj1-dj2)/(dj2*(dj2-1))
    
    return sj

def myCDF(jj,alpha,cdfOption):

    if (cdfOption == 'logCDF'):
        Fjj = st.logser.cdf(jj,alpha)
    elif (cdfOption == 'geomCDF'):
        Fjj = st.geom.cdf(jj,1-alpha)
    
    return Fjj

nCnt = 200
de = 3.0
do = 1.02
jjStart = 10
alpha1 = 0.75
alpha2 = 0.85
alpha3 = 0.95
alpha4 = 0.99

cdfOption = 'logCDF'  # geomCDF
# cdfOption = 'geomCDF'  # geomCDF

idx = [jj+1 for jj in range(nCnt)]


djj1 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha1,cdfOption)-myCDF(jjStart,alpha1,cdfOption)) \
                                    /(1-myCDF(jjStart,alpha1,cdfOption))) for jj in range(nCnt)]
djj2 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha2,cdfOption)-myCDF(jjStart,alpha2,cdfOption)) \
                                    /(1-myCDF(jjStart,alpha2,cdfOption))) for jj in range(nCnt)]
djj3 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha3,cdfOption)-myCDF(jjStart,alpha3,cdfOption)) \
                                    /(1-myCDF(jjStart,alpha3,cdfOption))) for jj in range(nCnt)]
djj4 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha4,cdfOption)-myCDF(jjStart,alpha4,cdfOption)) \
                                    /(1-myCDF(jjStart,alpha4,cdfOption))) for jj in range(nCnt)]

sk1 = [selCoeff_k(djj1[jj],djj1[jj+1]) for jj in range(nCnt)]
sk2 = [selCoeff_k(djj2[jj],djj2[jj+1]) for jj in range(nCnt)]
sk3 = [selCoeff_k(djj3[jj],djj3[jj+1]) for jj in range(nCnt)]
sk4 = [selCoeff_k(djj4[jj],djj4[jj+1]) for jj in range(nCnt)]    

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])

ax1.scatter(idx,np.log10(sk1),edgecolors='r',facecolors='none',label=r'$\alpha=0.75$')
ax1.scatter(idx,np.log10(sk2),edgecolors='g',facecolors='none',label=r'$\alpha=0.85$')
ax1.scatter(idx,np.log10(sk3),edgecolors='b',facecolors='none',label=r'$\alpha=0.95$')
ax1.scatter(idx,np.log10(sk4),edgecolors='k',facecolors='none',label=r'$\alpha=0.99$')

plt.xlim([0,nCnt])
plt.ylim([-15,1])

ax1.legend(loc='lower left')
plt.title('Diminishing Returns Epistastis')
plt.xlabel(r'$j^{th}$ Beneficial Mutation')
plt.ylabel(r'Selection Coeffcient $\log_{10}$')
plt.tight_layout()

ax2.scatter(idx,djj1[:-1],edgecolors='r',facecolors='none',label=r'$\alpha=0.75$')
ax2.scatter(idx,djj2[:-1],edgecolors='g',facecolors='none',label=r'$\alpha=0.85$')
ax2.scatter(idx,djj3[:-1],edgecolors='b',facecolors='none',label=r'$\alpha=0.95$')
ax2.scatter(idx,djj4[:-1],edgecolors='k',facecolors='none',label=r'$\alpha=0.99$')

plt.xlim([0,nCnt])
plt.ylim([1.0,3])

# plt.legend(loc='lower left')
plt.title('Diminishing Returns Epistastis')
plt.xlabel(r'$j^{th}$ Beneficial Mutation')
plt.ylabel(r'$d_j $- term')
plt.tight_layout()


# --------------- test pfix calculation


N = 3.73570525e+07
sc = 3.46731552e-05
pFix_c = 4.72663609e-05
Uc = 5.e-06


kk      = mcModel_rm.get_iExt()-1
N       = mcModel_rm.eq_Ni[kk]
sc      = mcModel_rm.sc_i[kk]
pFix_c  = mcModel_rm.pFix_c_i[kk]
Uc      = mcModel_rm.Uc_i[kk] 

# kk      = 1
# N       = mcModel_dre.eq_Ni[kk]
# sc      = mcModel_dre.sc_i[kk]
# pFix_c  = mcModel_dre.pFix_c_i[kk]
# Uc      = mcModel_dre.Uc_i[kk] 

MM_REGIME_MULTIPLE = 10
CI_TIMESCALE_TRANSITION = 0.5
DM_REGIME_MULTIPLE = 3

T_est = 1/(N*Uc*pFix_c)
T_swp = np.log(N*pFix_c)/sc

l_sU = np.log(sc/Uc)
l_Ns = np.log(N*sc)
l_Npfix = np.log(N*pFix_c)


# What regime?

if (sc <= 0) or (N <= 0) or (Uc <= 0) or (pFix_c <= 0):
    # bad evolutionary parameters
    regID = 0

# Calculate mean time between establishments
Test = 1/N*Uc*pFix_c

# Calculate mean time of sweep
Tswp = (1/sc)*np.log(N*pFix_c)
    
# calculate rate of adaptation based on regime
if (Test*CI_TIMESCALE_TRANSITION >= Tswp):
    # successional, establishment time scale exceeds sweep time scale
    regID = 1
    
elif (sc > MM_REGIME_MULTIPLE*Uc) and (Test <= CI_TIMESCALE_TRANSITION*Tswp):
    # multiple mutations, selection time scale smaller than  mutation time scale
    regID = 2
    
elif (Uc <= DM_REGIME_MULTIPLE*sc) and (Test <= CI_TIMESCALE_TRANSITION*Tswp):
    # diffusive mutations, 
    regID = 3
else:
    # regime undetermined
    regID = -1

regID


[N, sc, pFix_c, Uc, regID]


#%%

x1 = np.linspace(-1,1,21)
y1 = x1**2-0.5

x2 = np.linspace(-2,2,41)
y2 = x2**2-1

y1Sgn = np.sign(y1)
y2Sgn = np.sign(y2)


def calculate_v_intersections(vDiff):
    # calculate_v_intersections() determines points where the vDiff array cross
    # the zero axis, and also provide what type of crossing occurs.
    
    crossings   = []
    cross_types = []
    
    vDiffSgn = np.sign(vDiff)
    
    cross_1 = np.where(vDiffSgn                      == 0)[0]
    cross_2 = np.where(vDiffSgn[0:-1] + vDiffSgn[1:] == 0)[0]
    
    # check cross type 1 where v1 == v2
    for ii in range(len(cross_1)):
        idx = cross_1[ii]
        
        if idx == 0:
            crossings   = crossings   + [idx            ]
            cross_types = cross_types + [vDiffSgn[idx+1]]
            
        elif idx == len(vDiffSgn)-1:
            crossings   = crossings   + [ idx            ]
            cross_types = cross_types + [-vDiffSgn[idx-1]]
            
        else:
            if (vDiffSgn[idx-1] != vDiffSgn[idx+1]):
                crossSign   = vDiffSgn[idx+1] - vDiffSgn[idx-1]
                
                crossings   = crossings   + [idx       ]
                cross_types = cross_types + [crossSign ]
    
    # check cross type 2 where v1[ii] < v2[ii], v1[ii+1] > v2[ii+1]
    for ii in range(len(cross_2)):
    
        idx = cross_2[ii]
        
        if (idx == 0):
            minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
            
            crossings   = crossings   + [ idx + minIdx   ]
            cross_types = cross_types + [ vDiffSgn[idx+1]]
            
        elif (idx == len(vDiffSgn)-1):
            minIdx = np.argmin([vDiff[idx],vDiff[idx-1]])
            
            crossings   = crossings   + [ idx - minIdx   ]
            cross_types = cross_types + [-vDiffSgn[idx-1]]
            
        else:
            if (vDiffSgn[idx] != vDiffSgn[idx+1]):
                crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])
                minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
                
                crossings   = crossings   + [idx + minIdx ]
                cross_types = cross_types + [crossSign    ]
        
    return [crossings, cross_types]


[testCross,testCrossTypes] = calculate_v_intersections(y1)

[testCross,testCrossTypes] = calculate_v_intersections(y2)
# # -*- coding: utf-8 -*-
# """
# Created on Sat Jan 29 20:56:50 2022

# @author: Owner
# """

# plt.plot(sa_i,sr_i)
# plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

# plt.plot(Ua_i,Ur_i)
# plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

# plt.plot(pFixAbs_i,pFixRel_i)
# plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

# plt.plot(eq_yi,eq_Ni)
# plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))

# plt.scatter(state_i,va_i)
# plt.scatter(state_i,vr_i)
# plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))
# plt.ylim([0,1.5*max([max(va_i),max(vr_i)])])


# plt.scatter(di[1:],va_i)
# plt.scatter(di[1:],vr_i)
# plt.ticklabel_format(axis="both", style="sci",scilimits=(0,0))
# plt.ylim([0,1.5*max([max(va_i),max(vr_i)])])


# # pfix versus sel coeff
# fig,ax = plt.subplots(1,1,figsize=[7,7])
# ax.plot(mcModel1.state_i,mcModel1.pFix_c_i,color='red')
# ax.plot(mcModel1.state_i,mcModel1.sc_i,color='blue')

# fig,ax = plt.subplots(1,1,figsize=[7,7])
# ax.plot(mcModel1.state_i,mcModel1.pFix_d_i,color='red')
# ax.plot(mcModel1.state_i,mcModel1.sd_i,color='blue')

# # -----------------------------------------------------

# import matplotlib.pyplot as plt

# import os
# import sys
# sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

# from evoLibraries import evoObjects as evoObj
# from evoLibraries.MarkovChain import MC_RM_class as mcRM
# from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

# # The parameter file is read and a dictionary with their values is generated.
# paramFilePath = os.getcwd()+'/inputs/evoExp_RM_01_parameters.csv'
# modelType = 'RM'

# mcParams_rm = evoObj.evoOptions(paramFilePath, modelType)
# mcModel_rm = mcRM.mcEvoModel_RM(mcParams_rm.params)

# # The parameter file is read and a dictionary with their values is generated.
# paramFilePath = os.getcwd()+'/inputs/evoExp_DRE_01_parameters.csv'
# modelType = 'DRE'

# mcParams_dre = evoObj.evoOptions(paramFilePath, modelType)
# mcModel_dre = mcDRE.mcEvoModel_DRE(mcParams_dre.params)

# # unscaled rates

# fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])

# ax1.scatter(-mcModel_rm.di , mcModel_rm.get_vd_i_perUnitTime()   ,c='blue',label='vd')
# ax1.scatter(-mcModel_rm.di , mcModel_rm.get_vc_i_perUnitTime()   ,c='red',label='vc')
# ax1.plot(   -mcModel_rm.di , mcModel_rm.get_ve_i_perUnitTime()   ,c='black',label='ve')

# ax1.set_ylim(0,9e-4)

# ax2.scatter(-mcModel_dre.di, mcModel_dre.get_vd_i_perUnitTime()  ,c='blue',label='vd')
# ax2.scatter(-mcModel_dre.di, mcModel_dre.get_vc_i_perUnitTime()  ,c='red',label='vc')
# ax2.plot(   -mcModel_dre.di, mcModel_dre.get_ve_i_perUnitTime()  ,c='black',label='ve')

# ax2.set_ylim(0,9e-4)

# plt.legend()

# # evo regime

# fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])

# ax1.scatter(-mcModel_rm.di , mcModel_rm.evoRegime_d_i   ,c='blue',label='vd',marker = 'o')
# ax1.scatter(-mcModel_rm.di , mcModel_rm.evoRegime_c_i   ,c='red',label='vc' ,marker = '*')

# ax2.scatter(-mcModel_dre.di, mcModel_dre.evoRegime_d_i  ,c='blue',label='vd',marker = 'o')
# ax2.scatter(-mcModel_dre.di, mcModel_dre.evoRegime_c_i  ,c='red',label='vc' ,marker = '*')


# plt.legend()

# # -----------------------------------------------------
# # all three at once

# fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[7,10])
# ax1.plot(mcModel1.state_i,np.log10(mcModel1.sd_i),c='blue')
# ax1.plot(mcModel1.state_i,np.log10(mcModel1.sc_i),c='red')
# ax1.set_ylabel('s')

# ax2.plot(mcModel1.state_i,np.log10(mcModel1.pFix_d_i),c='blue')
# ax2.plot(mcModel1.state_i,np.log10(mcModel1.pFix_c_i),c='red')
# ax2.set_ylabel('$\pi_{fix}$')

# ax3.plot(mcModel1.state_i,np.log10(mcModel1.vd_i),c='blue')
# ax3.plot(mcModel1.state_i,np.log10(mcModel1.vc_i),c='red')
# ax3.set_ylabel('v')

# # -----------------------------------------------------
# # s and pfix

# fig,ax1 = plt.subplots(1,1,figsize=[7,5])
# ax1.plot(mcModel1.state_i,np.log10(mcModel1.sd_i),c='blue',label='sd')
# ax1.plot(mcModel1.state_i,np.log10(mcModel1.sc_i),c='red',label='sc')

# ax1.plot(mcModel1.state_i,np.log10(mcModel1.pFix_d_i),c='green',label='pFix d')
# ax1.plot(mcModel1.state_i,np.log10(mcModel1.pFix_c_i),c='purple',label='pFix c')
# ax1.set_ylabel(r'$s$ and $\pi_{fix}$')

# plt.legend()

# # -----------------------------------------------------
# # more general plots of quantities

# plt.plot(mcModel1.sd_i,mcModel1.sc_i)

# plt.scatter(mcModel1.state_i[0:100],np.log10(mcModel1.sd_i[0:100]))

# plt.plot(mcModel1.state_i[0:100],mcModel1.di[0:100]-1.02)

# plt.plot(mcModel1.state_i[0:100],np.log10(mcModel1.eq_Ni[0:100]))

# plt.plot(mcModel1.state_i[0:99],np.log10(np.abs(mcModel1.di[1:100]-mcModel1.di[0:99])/mcModel1.di[0:99]) )

# plt.plot(mcModel2.sd_i,mcModel2.sc_i)

# plt.plot(mcModel2.vd_i,mcModel2.vc_i)

# plt.scatter(mcModel2.Ud_i,mcModel2.Uc_i)

# # -----------------------------------------------

# fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
# ax1.scatter(mcModel_rm.state_i,mcModel_rm.di)
# ax2.scatter(mcModel_dre.state_i,mcModel_dre.di)

# fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
# ax1.scatter(mcModel_rm.di,mcModel_rm.sd_i)
# ax2.scatter(mcModel_dre.di,mcModel_dre.sd_i)

# # -----------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as st

# def dj_terms(do,de,Fjj):
    
#     dj = de*(do/de)**Fjj
    
#     return dj

# def selCoeff_k(dj1,dj2):
    
#     sj = (dj1-dj2)/(dj2*(dj2-1))
    
#     return sj

# def myCDF(jj,alpha,cdfOption):

#     if (cdfOption == 'logCDF'):
#         Fjj = st.logser.cdf(jj,alpha)
#     elif (cdfOption == 'geomCDF'):
#         Fjj = st.geom.cdf(jj,1-alpha)
    
#     return Fjj

# nCnt = 200
# de = 3.0
# do = 1.02
# jjStart = 10
# alpha1 = 0.75
# alpha2 = 0.85
# alpha3 = 0.95
# alpha4 = 0.99

# cdfOption = 'logCDF'  # geomCDF
# # cdfOption = 'geomCDF'  # geomCDF

# idx = [jj+1 for jj in range(nCnt)]


# djj1 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha1,cdfOption)-myCDF(jjStart,alpha1,cdfOption)) \
#                                     /(1-myCDF(jjStart,alpha1,cdfOption))) for jj in range(nCnt)]
# djj2 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha2,cdfOption)-myCDF(jjStart,alpha2,cdfOption)) \
#                                     /(1-myCDF(jjStart,alpha2,cdfOption))) for jj in range(nCnt)]
# djj3 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha3,cdfOption)-myCDF(jjStart,alpha3,cdfOption)) \
#                                     /(1-myCDF(jjStart,alpha3,cdfOption))) for jj in range(nCnt)]
# djj4 = [de] + [dj_terms(do,de,(myCDF(jj+jjStart+1,alpha4,cdfOption)-myCDF(jjStart,alpha4,cdfOption)) \
#                                     /(1-myCDF(jjStart,alpha4,cdfOption))) for jj in range(nCnt)]

# sk1 = [selCoeff_k(djj1[jj],djj1[jj+1]) for jj in range(nCnt)]
# sk2 = [selCoeff_k(djj2[jj],djj2[jj+1]) for jj in range(nCnt)]
# sk3 = [selCoeff_k(djj3[jj],djj3[jj+1]) for jj in range(nCnt)]
# sk4 = [selCoeff_k(djj4[jj],djj4[jj+1]) for jj in range(nCnt)]    

# fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,10])

# ax1.scatter(idx,np.log10(sk1),edgecolors='r',facecolors='none',label=r'$\alpha=0.75$')
# ax1.scatter(idx,np.log10(sk2),edgecolors='g',facecolors='none',label=r'$\alpha=0.85$')
# ax1.scatter(idx,np.log10(sk3),edgecolors='b',facecolors='none',label=r'$\alpha=0.95$')
# ax1.scatter(idx,np.log10(sk4),edgecolors='k',facecolors='none',label=r'$\alpha=0.99$')

# plt.xlim([0,nCnt])
# plt.ylim([-15,1])

# ax1.legend(loc='lower left')
# plt.title('Diminishing Returns Epistastis')
# plt.xlabel(r'$j^{th}$ Beneficial Mutation')
# plt.ylabel(r'Selection Coeffcient $\log_{10}$')
# plt.tight_layout()

# ax2.scatter(idx,djj1[:-1],edgecolors='r',facecolors='none',label=r'$\alpha=0.75$')
# ax2.scatter(idx,djj2[:-1],edgecolors='g',facecolors='none',label=r'$\alpha=0.85$')
# ax2.scatter(idx,djj3[:-1],edgecolors='b',facecolors='none',label=r'$\alpha=0.95$')
# ax2.scatter(idx,djj4[:-1],edgecolors='k',facecolors='none',label=r'$\alpha=0.99$')

# plt.xlim([0,nCnt])
# plt.ylim([1.0,3])

# # plt.legend(loc='lower left')
# plt.title('Diminishing Returns Epistastis')
# plt.xlabel(r'$j^{th}$ Beneficial Mutation')
# plt.ylabel(r'$d_j $- term')
# plt.tight_layout()


# # --------------- test pfix calculation


# N = 3.73570525e+07
# sc = 3.46731552e-05
# pFix_c = 4.72663609e-05
# Uc = 5.e-06


# kk      = mcModel_rm.get_iExt()-1
# N       = mcModel_rm.eq_Ni[kk]
# sc      = mcModel_rm.sc_i[kk]
# pFix_c  = mcModel_rm.pFix_c_i[kk]
# Uc      = mcModel_rm.Uc_i[kk] 

# # kk      = 1
# # N       = mcModel_dre.eq_Ni[kk]
# # sc      = mcModel_dre.sc_i[kk]
# # pFix_c  = mcModel_dre.pFix_c_i[kk]
# # Uc      = mcModel_dre.Uc_i[kk] 

# MM_REGIME_MULTIPLE = 10
# CI_TIMESCALE_TRANSITION = 0.5
# DM_REGIME_MULTIPLE = 3

# T_est = 1/(N*Uc*pFix_c)
# T_swp = np.log(N*pFix_c)/sc

# l_sU = np.log(sc/Uc)
# l_Ns = np.log(N*sc)
# l_Npfix = np.log(N*pFix_c)


# # What regime?

# if (sc <= 0) or (N <= 0) or (Uc <= 0) or (pFix_c <= 0):
#     # bad evolutionary parameters
#     regID = 0

# # Calculate mean time between establishments
# Test = 1/N*Uc*pFix_c

# # Calculate mean time of sweep
# Tswp = (1/sc)*np.log(N*pFix_c)
    
# # calculate rate of adaptation based on regime
# if (Test*CI_TIMESCALE_TRANSITION >= Tswp):
#     # successional, establishment time scale exceeds sweep time scale
#     regID = 1
    
# elif (sc > MM_REGIME_MULTIPLE*Uc) and (Test <= CI_TIMESCALE_TRANSITION*Tswp):
#     # multiple mutations, selection time scale smaller than  mutation time scale
#     regID = 2
    
# elif (Uc <= DM_REGIME_MULTIPLE*sc) and (Test <= CI_TIMESCALE_TRANSITION*Tswp):
#     # diffusive mutations, 
#     regID = 3
# else:
#     # regime undetermined
#     regID = -1

# regID


# [N, sc, pFix_c, Uc, regID]


# #%%

# x1 = np.linspace(-1,1,21)
# y1 = x1**2-0.5

# x2 = np.linspace(-2,2,41)
# y2 = x2**2-1

# y1Sgn = np.sign(y1)
# y2Sgn = np.sign(y2)


# def calculate_v_intersections(vDiff):
#     # calculate_v_intersections() determines points where the vDiff array cross
#     # the zero axis, and also provide what type of crossing occurs.
    
#     crossings   = []
#     cross_types = []
    
#     vDiffSgn = np.sign(vDiff)
    
#     cross_1 = np.where(vDiffSgn                      == 0)[0]
#     cross_2 = np.where(vDiffSgn[0:-1] + vDiffSgn[1:] == 0)[0]
    
#     # check cross type 1 where v1 == v2
#     for ii in range(len(cross_1)):
#         idx = cross_1[ii]
        
#         if idx == 0:
#             crossings   = crossings   + [idx            ]
#             cross_types = cross_types + [vDiffSgn[idx+1]]
            
#         elif idx == len(vDiffSgn)-1:
#             crossings   = crossings   + [ idx            ]
#             cross_types = cross_types + [-vDiffSgn[idx-1]]
            
#         else:
#             if (vDiffSgn[idx-1] != vDiffSgn[idx+1]):
#                 crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])
                
#                 crossings   = crossings   + [idx       ]
#                 cross_types = cross_types + [crossSign ]
    
#     # check cross type 2 where v1[ii] < v2[ii], v1[ii+1] > v2[ii+1]
#     for ii in range(len(cross_2)):
    
#         idx = cross_2[ii]
        
#         if (idx == 0):
#             minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
            
#             crossings   = crossings   + [ idx + minIdx   ]
#             cross_types = cross_types + [ vDiffSgn[idx+1]]
            
#         elif (idx == len(vDiffSgn)-1):
#             minIdx = np.argmin([vDiff[idx],vDiff[idx-1]])
            
#             crossings   = crossings   + [ idx - minIdx   ]
#             cross_types = cross_types + [-vDiffSgn[idx-1]]
            
#         else:
#             if (vDiffSgn[idx] != vDiffSgn[idx+1]):
#                 crossSign   = np.sign(vDiffSgn[idx+1] - vDiffSgn[idx-1])
#                 minIdx = np.argmin([vDiff[idx],vDiff[idx+1]])
                
#                 crossings   = crossings   + [idx + minIdx ]
#                 cross_types = cross_types + [crossSign    ]
        
#     return [crossings, cross_types]


# [testCross,testCrossTypes] = calculate_v_intersections(y1)
# testCross
# testCrossTypes
# y1[testCross[0]]
# y1[testCross[1]]

# [testCross,testCrossTypes] = calculate_v_intersections(y2)
# testCross
# testCrossTypes
# y2[testCross[0]]
# y2[testCross[1]]


# fid = open('test.txt','w')
# fid.write(str(v_cross_idx))
# fid.write('\n')
# fid.write(str(attract_cross_idxs))
# fid.write('\n')
# fid.write(str(v_cross_types))
# fid.write('\n')
# fid.write(str(len(idx_map)))
# fid.close()


import matplotlib.pyplot as plt
import numpy as np

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

[ss,sst]=mcModel_rm.get_v_intersect_state_index(mcModel_rm.vc_i)

fig,ax = plt.subplots(1,1,figsize=[7,7])
ax.plot(mcModel_rm.state_i,np.log10(mcModel_rm.vd_i))
ax.plot(mcModel_rm.state_i,np.log10(mcModel_rm.vc_i))
ax.scatter(mcModel_rm.state_i[ss],np.log10(mcModel_rm.vd_i[ss]))

#%%----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

ii=1
jj=13

mcTestParams = mcModels.get_params_ij(ii,jj)
# mcTestParams2 = mcModels.get_params_ij(0,0)
# mcTestParams3 = mcModels.get_params_ij(2,2)

mcTestModel = mcRM.mcEvoModel_RM(mcTestParams)
mcTestEqParams = mcTestModel.get_stable_state_evo_parameters()


# testParamGrid = mcModels.get_evoParam_grid('UdMax',0)

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


fig2,(ax21,ax22) = plt.subplots(2,1,figsize=[7,12])
ax21.scatter(mcTestModel.state_i,np.log10(mcTestModel.vd_i),label='vd')
ax21.scatter(mcTestModel.state_i,np.log10(mcTestModel.vc_i),label='vc')
ax21.legend()

params_stable_state = mcTestModel.get_stable_state_evo_parameters()

ax21.text(-120,-1,params_stable_state['eqState'], fontsize = 22)

ax22.scatter(mcTestModel.state_i,mcTestModel.evoRegime_d_i,label='reg_d')
ax22.scatter(mcTestModel.state_i,mcTestModel.evoRegime_c_i,label='reg_c')
ax22.legend()


# fig3,(ax31,ax32,ax33,ax34) = plt.subplots(4,1,figsize=[7,12])
# ax31.scatter(mcTestModel.state_i,np.log10(mcTestModel.sd_i),label='sd')
# ax31.scatter(mcTestModel.state_i,np.log10(mcTestModel.sc_i),label='sc')
# ax31.legend()

# params_stable_state = mcTestModel.get_stable_state_evo_parameters()
# rho_val = mcTestModel.calculate_evoRho()

# ax31.text(-120,-1,params_stable_state['eqState'], fontsize = 22)
# ax31.text(-120,-2,rho_val, fontsize = 22)

# ax32.scatter(mcTestModel.state_i,np.log10(mcTestModel.Ud_i),label='Ud')
# ax32.scatter(mcTestModel.state_i,np.log10(mcTestModel.Uc_i),label='Uc')
# ax32.legend()
# ax33.scatter(mcTestModel.state_i,np.log10(mcTestModel.pFix_d_i),label='pFix_d')
# ax33.scatter(mcTestModel.state_i,np.log10(mcTestModel.pFix_c_i),label='pFix_c')
# ax33.legend()
# ax34.scatter(mcTestModel.state_i,np.log10(mcTestModel.eq_Ni),label='pFix_c')
# ax34.legend()

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.flatten(),Y.flatten(),Z.flatten()) 
ax.set_xlabel('s ratio')
ax.set_ylabel('U ratio')
ax.set_zlabel('rho')

#%%

# -----------------------------------------------------------------------

rho_ij = array([
[2.08783283e-03, 8.39203396e-03, 3.66313113e-02, 1.69041160e-01, 7.45857562e-01, 9.10608482e-01, 1.00620538e+00, 1.01774314e+00, 5.54342700e+00, 8.15484999e+01, 4.92875856e+02], 
[1.87264059e-03, 7.52706977e-03, 3.28557341e-02, 1.51618143e-01, 7.26657319e-01, 9.28324552e-01, 1.00882643e+00, 1.04105520e+00, 4.74464659e+00, 6.98857056e+01, 4.23212830e+02],
[1.66914945e-03, 6.70913813e-03, 2.92854544e-02, 1.35142505e-01, 6.47694850e-01, 9.64630076e-01, 1.05183315e+00, 1.20725735e+00, 4.00797835e+00, 5.91224642e+01, 3.58854938e+02],
[1.47735942e-03, 5.93823904e-03, 2.59204722e-02, 1.19614246e-01, 5.73272866e-01, 9.79882918e-01, 1.05226173e+00, 1.15706862e+00, 3.33342229e+00, 4.92587755e+01, 2.99802179e+02],
[1.29727050e-03, 5.21437251e-03, 2.27607876e-02, 1.05033366e-01, 5.03391368e-01, 9.95716962e-01, 1.06371155e+00, 1.22038887e+00, 2.72097840e+00, 4.02946398e+01, 2.46054556e+02],
[1.12888269e-03, 4.53753852e-03, 1.98064006e-02, 9.13998653e-02, 4.38050354e-01, 1.01663722e+00, 1.10831279e+00, 1.22066100e+00, 2.17064669e+00, 3.22300568e+01, 1.97612066e+02],
[9.72195993e-04, 3.90773709e-03, 1.70573111e-02, 7.87137435e-02, 3.77249825e-01, 1.02812796e+00, 1.10734836e+00, 1.15792435e+00, 1.68242715e+00, 2.50650267e+01, 1.54474710e+02],
[8.27210402e-04, 3.32496821e-03, 1.45135192e-02, 6.69750008e-02, 3.20989782e-01, 1.02130428e+00, 1.13018415e+00, 1.22123385e+00, 1.46904528e+00, 1.87995495e+01, 1.16642488e+02],
[6.93925920e-04, 2.78923188e-03, 1.21750248e-02, 5.61836371e-02, 2.69270223e-01, 1.04311789e+00, 1.15015157e+00, 1.23346082e+00, 1.38381946e+00, 1.34336250e+01, 8.41154009e+01],
[5.72342548e-04, 2.30052810e-03, 1.00418279e-02, 4.63396525e-02, 2.22091150e-01, 1.08811904e+00, 1.15706536e+00, 1.20923545e+00, 1.46452078e+00, 8.96725346e+00, 5.68934476e+01],
[4.62460284e-04, 1.85885687e-03, 8.11392866e-03, 3.74430469e-02, 1.79452562e-01, 8.79214452e-01, 1.17762411e+00, 1.23464045e+00, 1.38666685e+00, 5.40043472e+00, 3.49766284e+01]])



#%% Numerical test for improvements to calculation of densities



vDiff = np.array([np.nan, 0.00123322, 0.00127919, 0.00130212, 0.00131523, 0.00132296])

from evoLibraries.MarkovChain import MC_RM_class as mcRM

testParams = mcModels.get_params_ij(0,0)

testMC = mcRM.mcEvoModel_RM(testParams)
 
fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[7,12])
ax1.scatter(testMC.state_i,np.log10(testMC.vd_i),label='vd')
ax1.scatter(testMC.state_i,np.log10(testMC.vc_i),label='vc')
ax1.legend()
ax2.scatter(testMC.state_i,np.log10(testMC.sd_i),label='sd')
ax2.scatter(testMC.state_i,np.log10(testMC.pFix_d_i),label='pFix_d')
ax2.legend()
ax3.scatter(testMC.state_i,np.log10(testMC.sc_i),label='sc')
ax3.scatter(testMC.state_i,np.log10(testMC.pFix_c_i),label='pFix_c')
ax3.legend()

fig2,ax21 = plt.subplots(1,1,figsize=[7,7])
ax21.scatter(testMC.state_i,np.log10(testMC.eq_Ni),label='eqNi')
ax21.legend()

#%% matrix solution test code

import random
from scipy.linalg import null_space
from scipy.linalg import solve

A = np.random.random((3,3))

A = np.array([[1,0,0,0], \
     [0.01,0.40,0.30,0.29], \
     [0.00,0.11,0.50,0.39], \
     [0.00,0.00,0.00,1.00]])

A = np.transpose(A)
    
# for jj in range(A.shape[0]):
#     colSum = sum(A[:,jj])
#     for ii in range(A.shape[1]):
#         A[ii,jj] = A[ii,jj]/colSum

B = A[1:3,1:3] 
b_fix = np.array(A[3,1:3])
b_ext = np.array(A[0,1:3])

# Z = null_space(A-np.eye(A.shape[0]))

# xSol = Z/sum(Z[:,0])

xSol_fix = solve(np.eye(B.shape[0])-B.transpose(),b_fix)    
xSol_ext = solve(np.eye(B.shape[0])-B.transpose(),b_ext)    

# u = np.array([[0.2],[0.3],[0.1],[0.4]])

# np.matmul(A,u)
# np.matmul(A.transpose(),u)
    
# np.matmul(np.ones([1,4]),np.matmul(A,u))
# np.matmul(np.ones([1,4]),np.matmul(A.transpose(),u))


#%%


# b - birth term
# T - Territory size
# d - array with death terms
# c - array with competition terms
# n1 - max array size of transition probabilities (competition and death phase)
# n2 - max array size of transition probabilities (solve for pFix)
    
# set sim parameters                                        
# d           = np.array([di[iCheck+1]  , di[iCheck-15]])
# d           = np.array([2.388760  ,  2.3270015947896763])
# d           = np.array([2.388760  ,  2.2270015947896763])
from evoLibraries.LotteryModel import LM_pFix_MA as lmPfixMa
from evoLibraries.LotteryModel import LM_pFix_FSA as lmPfixFsa

b           = 2
d           = np.array([2.388760  ,  2.388760])
cp          = 0.04
c           = np.array([1         ,  1+    cp])
T           = 1e9
n1          = 1000
n2          = 500
pfix_option = 0

[Tc,Td,Ts, Ts_solve, pjj_solve,pFix] = lmPfixMa.calc_pFix_MA(b,T,d,c,n1,n2,0)

pjj_solve = np.transpose(Ts[-1,1:n2-1])
pFix_ii = solve(Ts_solve, pjj_solve)

pFix = pFix_ii[0] 



#%%



for ii in [0,2]:
    print(ii)













for ii in range(5):
    for jj in range(5):
        mTest[ii,jj]=ii+jj
        
        
a = [[1,1],[0,1]] 
b = [[4,1],[2,2]] 
print (np.matmul(a,b))




#%%

# TEST PLOTS FOR EVO GRID WITH DRE

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

ii=0
jj=10

mcTestEvoOptions = mcModels.get_params_ij(ii,jj)
# mcTestParams2 = mcModels.get_params_ij(0,0)
# mcTestParams3 = mcModels.get_params_ij(2,2)

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


fig2,(ax21,ax22) = plt.subplots(2,1,figsize=[7,12])
ax21.scatter(mcTestModel.state_i,np.log10(mcTestModel.vd_i),label='vd')
ax21.scatter(mcTestModel.state_i,np.log10(mcTestModel.vc_i),label='vc')
ax21.legend()

params_stable_state = mcTestModel.get_stable_state_evo_parameters()

ax21.text(0,-4,params_stable_state['eqState'], fontsize = 22)

ax22.scatter(mcTestModel.state_i,mcTestModel.evoRegime_d_i,label='reg_d')
ax22.scatter(mcTestModel.state_i,mcTestModel.evoRegime_c_i,label='reg_c')
ax22.legend()


# fig3,(ax31,ax32,ax33,ax34) = plt.subplots(4,1,figsize=[7,12])
# ax31.scatter(mcTestModel.state_i,np.log10(mcTestModel.sd_i),label='sd')
# ax31.scatter(mcTestModel.state_i,np.log10(mcTestModel.sc_i),label='sc')
# ax31.legend()

# params_stable_state = mcTestModel.get_stable_state_evo_parameters()
# rho_val = mcTestModel.calculate_evoRho()

# ax31.text(-120,-1,params_stable_state['eqState'], fontsize = 22)
# ax31.text(-120,-2,rho_val, fontsize = 22)

# ax32.scatter(mcTestModel.state_i,np.log10(mcTestModel.Ud_i),label='Ud')
# ax32.scatter(mcTestModel.state_i,np.log10(mcTestModel.Uc_i),label='Uc')
# ax32.legend()
# ax33.scatter(mcTestModel.state_i,np.log10(mcTestModel.pFix_d_i),label='pFix_d')
# ax33.scatter(mcTestModel.state_i,np.log10(mcTestModel.pFix_c_i),label='pFix_c')
# ax33.legend()
# ax34.scatter(mcTestModel.state_i,np.log10(mcTestModel.eq_Ni),label='pFix_c')
# ax34.legend()

#%%

# SCRIPTS TO FINE TUNE THE MATRIX APPROACH FOR PFIX

# pFix d-trait beneficial mutation


import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d\\evoLibraries')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_RM_class as mcRM

import evoLibraries.LotteryModel.LM_pFix_FSA as lmPfix
import evoLibraries.LotteryModel.LM_pFix_MA as lmPfixMa

# The parameter file is read and a dictionary with their values is generated.
paramFilePath = os.getcwd()+'/inputs/evoExp_RM_03_parameters.csv'
# paramFilePath = os.getcwd()+'/inputs/evoExp_RM_04_parameters.csv'
modelType = 'RM'

mcParams = evoObj.evoOptions(paramFilePath, modelType)
mcModel = mcRM.mcEvoModel_RM(mcParams.params)

dArry = np.array( [mcModel.di[ii], mcModel.di[ii+1] ] )
cArry = np.array( [1, 1] )

# Test entry of chain
ii = 10
kMax = 10
pFix_FSA_val = lmPfix.calc_pFix_FSA(mcParams.params['b'], mcParams.params['T'], dArry, cArry, kMax)

n1 = 100
n2 = 100
pfix_option = 1
[Tc,Td,Ts, Ts_solve, pjj_solve, pFix_MA_val] = lmPfixMa.calc_pFix_MA(mcParams.params['b'], mcParams.params['T'], dArry, cArry,n1,n2,pfix_option)


# plot the pfix values of the mc chain
plt.plot(mcModel.state_i,mcModel.pFix_d_i)

#%%

ii = 0
jj = 3
mcModels.get_evoParam_grid('Ua',0)[ii,jj]
mcModels.get_evoParam_grid('cp',1)[ii,jj]



#%%


# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, 'D:\\Documents\\GitHub\\compEvo2d')

from evoLibraries import evoObjects as evoObj
from evoLibraries.MarkovChain import MC_DRE_class as mcDRE

from matplotlib.lines import Line2D

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
paramFilePath1 = os.getcwd()+'/inputs/evoExp_DRE_bEvo_04_parameters.csv'
modelType = 'DRE'
absFitType = 'bEvo'

mcParams = evoObj.evoOptions(paramFilePath1, modelType, absFitType)


T_vals = [1e5,1e7,1e10]
myColors = ["blue","lime","cyan"]
T_vals_strVd = [r'$v_d (T=10^5)$',r'$v_d (T=10^7)$',r'$v_d (T=10^{10})$']
T_vals_strVc = [r'$v_c (T=10^5)$',r'$v_c (T=10^7)$',r'$v_c (T=10^{10})$']
T_vals_strLgd = [r'$T=10^5$',r'$T=10^7$',r'$T=10^{10}$']
T_select = [1,2]

mcModels = []

for ii in T_select:
    mcParams.params['T'] = T_vals[ii]
    mcModels = mcModels + [ mcDRE.mcEvoModel_DRE(mcParams) ]


# --------------------------------------------------------------------------
#                               Figure - Panel (A)
# --------------------------------------------------------------------------
fig1, ax1 = plt.subplots(1,1,figsize=[7,6])

#ax1.plot(state1_i,ve1_i,color="black",linewidth=2,label=r'$v_e$')
#ax1.plot(eq_y1i,va1_i,color="green",linewidth=2,label=r'$v_{a,1}$')
#ax1.plot(eq_y1i,vr1_i,'-.',color="green",linewidth=2,label=r'$v_{r,1}$')
scaleFactor = 1e4

for ii in range(len(mcModels)):
    ax1.plot(mcModels[ii].state_i,mcModels[ii].va_i*scaleFactor,color=myColors[ii],linewidth=2,label=T_vals_strVd[ii])
    ax1.plot(mcModels[ii].state_i,mcModels[ii].vc_i*scaleFactor,'-.',color=myColors[ii],linewidth=2,label=T_vals_strVc[ii])

# equilibrium states and equilibrium rates of adaptation
idx = [mcModels[ii].get_mc_stable_state_idx() for ii in range(len(mcModels))]
iEq = np.asarray([mcModels[ii].state_i[idx[ii]-1] for ii in range(len(mcModels))])
vEq = np.asarray([mcModels[ii].va_i[idx[ii]-1] for ii in range(len(mcModels))])

for ii in range(len(mcModels)):
    ax1.scatter(iEq[ii],vEq[ii]*scaleFactor,marker="o",s=40,c="black")

for ii in range(len(mcModels)):
    ax1.plot([iEq[ii],iEq[ii]],[0,vEq[ii]*scaleFactor],c="black",linewidth=1,linestyle=':')

# axes and label adjustements
# ax1.set_xlim(0,1)
xTickMax = int(mcModels[0].get_iMax()/25+1)

xBack = 3
ax1.set_xlim(50,25*(xTickMax-1-xBack))    
ax1.set_xticks([25*i for i in range(2,xTickMax-xBack)])
ax1.set_xticklabels([str(25*i) for i in range(2,xTickMax-xBack)],fontsize=16)
ax1.set_yticks([0.2*i for i in range(0,7)])
ax1.set_yticklabels([str(2*i/10.0) for i in range(0,7)],fontsize=16)
ax1.set_ylim(0,1.4)    # 2,5e04 ~ 1.5*max([max(va_i),max(vr_i)])
ax1.set_xlabel(r'Absolute Fitness Class',fontsize=20,labelpad=8)
ax1.set_ylabel(r'Rate of adaptation',fontsize=20,labelpad=8)

arrwLngth1 = 4
arrwOffset = -2
vScale = 0.1
ax1.annotate("", xy=(iEq[0]-arrwLngth1-arrwOffset,vScale*vEq[ii]*scaleFactor), xytext=(iEq[0]-arrwOffset,vScale*vEq[ii]*scaleFactor),arrowprops={'arrowstyle':'->','lw':2})

custom_lines = [Line2D([0], [0], color=myColors[ii], lw=2) for ii in range(len(mcModels))]
ax1.legend(custom_lines,[ T_vals_strLgd[ii] for ii in T_select],fontsize=14)

# save figure
plt.tight_layout()
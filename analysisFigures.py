# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:28:52 2022

@author: dirge
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import evo_library as myfun            # my functions in a seperate file
from mpl_toolkits.mplot3d import Axes3D
import copy as cpy
import pickle 
import matplotlib.tri as tri
from mpl_toolkits import mplot3d

# --------------------------------------------------------------------------
#                               Load data
# --------------------------------------------------------------------------

# load the array data for rho and variable parameter
with open('outputs/fig_rho_UvsS_data.pickle', 'rb') as f:
    [P1_ARRY, P2_ARRY, RHO_ARRY, Y_ARRY, P1_valMid, P2_valMid, effSa_ARRY, effUa_ARRY, effSr_ARRY, effUr_ARRY, paramsTemp, dMax] = pickle.load(f)


# Goal is to generate a contour plot of rho over ranges of the effective parameters

nPts = int(effSr_ARRY.shape[0])*int(effSr_ARRY.shape[1])

#--------------------------------------------------------
x = np.reshape(P2_ARRY,nPts)
y = np.reshape(P1_ARRY,nPts)

plt.scatter(np.log10(x),np.log10(y),c='r')
plt.xlabel('cr')
plt.ylabel('Ua_max')

# -------------------------------------------------------
x = np.reshape(effSr_ARRY,nPts)
y = np.reshape(effUa_ARRY,nPts)
z = np.reshape(Y_ARRY,nPts)

plt.scatter(np.log10(x),np.log10(y),c=z)
plt.colorbar()
plt.xlabel('eff sr')
plt.ylabel('eff Ua')
plt.title('pop density')

# -------------------------------------------------------
x = np.reshape(effSr_ARRY,nPts)
y = np.reshape(effUa_ARRY,nPts)
z = np.reshape(RHO_ARRY,nPts)

plt.scatter(np.log10(x),np.log10(y),c=z)
plt.colorbar()
plt.xlabel('eff sr')
plt.ylabel('eff Ua')
plt.title('rho')
#--------------------------------------------------------
x = np.reshape(P2_ARRY,nPts)
y = np.reshape(effSr_ARRY,nPts)


plt.scatter(np.log10(x),np.log10(y),c='r')
plt.xlabel('cr')
plt.ylabel('eff sr')

# -------------------------------------------------------
# Max absolute mutation rate vs eff mutation rate

x = np.reshape(P1_ARRY,nPts)
y = np.reshape(effUa_ARRY,nPts)
z = np.reshape(Y_ARRY,nPts)

plt.scatter(np.log10(x),np.log10(y),c=z)
plt.xlabel('Ua_max')
plt.ylabel('eff Ua')
plt.colorbar()

# -------------------------------------------------------

x = np.reshape(P2_ARRY,nPts)
y = np.reshape(effUa_ARRY,nPts)

plt.scatter(np.log10(x),np.log10(y),c='r')
plt.xlabel('cr')
plt.ylabel('eff Ua')

# -------------------------------------------------------

x = np.reshape(P1_ARRY,nPts)
y = np.reshape(effSr_ARRY,nPts)

plt.scatter(np.log10(x),np.log10(y),c='r')
plt.xlabel('Ua_max')
plt.ylabel('eff Sr')

# -------------------------------------------------------

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.reshape(np.log10(effSr_ARRY),nPts)
y = np.reshape(np.log10(effUa_ARRY),nPts)
z = np.reshape(RHO_ARRY,nPts)

ax.scatter(x, y, z)
plt.show()


# --------------------------------------------------------------
# Matrix plots for joanna showing the relationship between Model 
# parameters and evolutionary parameters. Dots show color gradient
# of population density (version 01)

cr = np.reshape(P2_ARRY,nPts)
UaMax = np.reshape(P1_ARRY,nPts)
sr = np.reshape(effSr_ARRY,nPts)
Ua = np.reshape(effUa_ARRY,nPts)
y = np.reshape(Y_ARRY,nPts)

#myLabels = [['UaMax','sr'],['cr','sr'],['UaMax','Ua'],['cr','Ua']]
myLabels = [[' ','sr'],[' ',' '],['UaMax','Ua'],['cr',' ']]

fig, axs = plt.subplots(2, 2)

fig.suptitle('Effective & Model Parameters (pop density shown in colorbar)\nParameter shown on log10 Scale\nUr = 1e-5 and sa = 1e-2')

p1 = axs[0,0].scatter(np.log10(UaMax),  np.log10(sr),c=y)
p2 = axs[0,1].scatter(np.log10(cr),     np.log10(sr),c=y)
p3 = axs[1,0].scatter(np.log10(UaMax),  np.log10(Ua),c=y)
p4 = axs[1,1].scatter(np.log10(cr),     np.log10(Ua),c=y)

for (ii,ax) in enumerate(axs.flat):
    ax.set(xlabel = myLabels[ii][0],ylabel = myLabels[ii][1])

#for ax in axs.flat:
#    ax.label_outer()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(p1,cax=cbar_ax)

# --------------------------------------------------------------
# Matrix plots for joanna showing the relationship between Model 
# parameters and evolutionary parameters. Dots show color gradient
# of population density (version 02)

cr = np.reshape(P2_ARRY,nPts)
UaMax = np.reshape(P1_ARRY,nPts)
sr = np.reshape(effSr_ARRY,nPts)
Ua = np.reshape(effUa_ARRY,nPts)
y = np.reshape(Y_ARRY,nPts)

#myLabels = [['UaMax','cr'],['sr','cr'],['UaMax','Ua'],['sr','Ua']]
myLabels = [[' ','cr'],[' ',' '],['UaMax','Ua'],['sr',' ']]

fig, axs = plt.subplots(2, 2)

fig.suptitle('Effective & Model Parameters (pop density shown in colorbar)\nParameters shown on log10 Scale\nUr = 1e-5 and sa = 1e-2')

p1 = axs[0,0].scatter(np.log10(UaMax),  np.log10(cr),c=y)
p2 = axs[0,1].scatter(np.log10(sr),     np.log10(cr),c=y)
p3 = axs[1,0].scatter(np.log10(UaMax),  np.log10(Ua),c=y)
p4 = axs[1,1].scatter(np.log10(sr),     np.log10(Ua),c=y)

for (ii,ax) in enumerate(axs.flat):
    ax.set(xlabel = myLabels[ii][0],ylabel = myLabels[ii][1])


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(p1,cax=cbar_ax)


# --------------------------------------------------------------
# Matrix plots for joanna showing the relationship between Model 
# parameters and evolutionary parameters. Dots show color gradient
# of population density (version 02)

cr = np.reshape(P2_ARRY,nPts)
UaMax = np.reshape(P1_ARRY,nPts)
sr = np.reshape(effSr_ARRY,nPts)
Ua = np.reshape(effUa_ARRY,nPts)
rho = np.reshape(RHO_ARRY,nPts)

#myLabels = [['UaMax','cr'],['sr','cr'],['UaMax','Ua'],['sr','Ua']]
myLabels = [[' ','cr'],[' ',' '],['UaMax','Ua'],['sr',' ']]

fig, axs = plt.subplots(2, 2)

fig.suptitle('Effective & Model Parameters (rho shown in colorbar)\nParameters shown on log10 Scale\nUr = 1e-5 and sa = 1e-2')

p1 = axs[0,0].scatter(np.log10(UaMax),  np.log10(cr),c=rho)
p2 = axs[0,1].scatter(np.log10(sr),     np.log10(cr),c=rho)
p3 = axs[1,0].scatter(np.log10(UaMax),  np.log10(Ua),c=rho)
p4 = axs[1,1].scatter(np.log10(sr),     np.log10(Ua),c=rho)

for (ii,ax) in enumerate(axs.flat):
    ax.set(xlabel = myLabels[ii][0],ylabel = myLabels[ii][1])


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(p1,cax=cbar_ax)

# ------------------------------------------------------------------
# 3d plot UaMax vs Ua vs Rho

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(np.log10(UaMax), np.log10(Ua), rho)
plt.title('UaMax vs Ua vs rho')
plt.xlabel('UaMax')
plt.ylabel('Ua')
plt.show()




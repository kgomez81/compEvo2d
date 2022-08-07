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
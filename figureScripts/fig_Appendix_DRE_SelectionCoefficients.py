# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:59:37 2022

@author: dirge
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logser

def dj_terms(do,de,Fjj):
    
    dj = de*(do/de)**Fjj
    
    return dj

def selCoeff_k(dj1,dj2):
    
    sj = (dj1-dj2)/(dj2*(dj2-1))
    
    return sj

nCnt = 100
de = 3.1
do = 1.1
alpha1 = 0.75
alpha2 = 0.85
alpha3 = 0.95
alpha4 = 0.99

idx = [jj+1 for jj in range(nCnt)]

djj1 = [de] + [dj_terms(do,de,logser.cdf(jj+1,alpha1)) for jj in range(nCnt)]
djj2 = [de] + [dj_terms(do,de,logser.cdf(jj+1,alpha2)) for jj in range(nCnt)]
djj3 = [de] + [dj_terms(do,de,logser.cdf(jj+1,alpha3)) for jj in range(nCnt)]
djj4 = [de] + [dj_terms(do,de,logser.cdf(jj+1,alpha4)) for jj in range(nCnt)]

sk1 = [selCoeff_k(djj1[jj],djj1[jj+1]) for jj in range(nCnt)]
sk2 = [selCoeff_k(djj2[jj],djj2[jj+1]) for jj in range(nCnt)]
sk3 = [selCoeff_k(djj3[jj],djj3[jj+1]) for jj in range(nCnt)]
sk4 = [selCoeff_k(djj4[jj],djj4[jj+1]) for jj in range(nCnt)]    

fig,ax = plt.subplots(1,1,figsize=[7,5])

#ax.scatter(idx,np.log10(sk1),c='r')
#ax.scatter(idx,np.log10(sk2),c='g')
#ax.scatter(idx,np.log10(sk3),c='b')
#ax.scatter(idx,np.log10(sk4),c='k')

ax.scatter(idx,np.log10(sk1),edgecolors='r',facecolors='none',label=r'$\alpha=0.75$')
ax.scatter(idx,np.log10(sk2),edgecolors='g',facecolors='none',label=r'$\alpha=0.85$')
ax.scatter(idx,np.log10(sk3),edgecolors='b',facecolors='none',label=r'$\alpha=0.95$')
ax.scatter(idx,np.log10(sk4),edgecolors='k',facecolors='none',label=r'$\alpha=0.99$')

plt.xlim([0,100])
plt.ylim([-15,1])

plt.legend(loc='lower left')
plt.title('Diminishing Returns Epistastis')
plt.xlabel(r'$j^{th}$ Beneficial Mutation')
plt.ylabel(r'Selection Coeffcient $\log_{10}$')
plt.tight_layout()

fig.savefig('figures/Appendix/fig_Appendix_DRE_SelectionCoefficients.pdf')
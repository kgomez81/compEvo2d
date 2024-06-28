# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:26:48 2024

@author: Kevin Gomez
"""
# --------------------------------------------------------------------------
#                               Libraries   
# --------------------------------------------------------------------------
import numpy as np

# --------------------------------------------------------------------------
#                               Functions   
# --------------------------------------------------------------------------

def getScatterData(X,Y,Z):
    
    x = []
    y = []
    z = []
    
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            
            # removed bad data
            xGood = not np.isnan(X[ii,jj]) and not np.isinf(X[ii,jj])
            yGood = not np.isnan(Y[ii,jj]) and not np.isinf(Y[ii,jj])
            zGood = not np.isnan(Z[ii,jj]) and not np.isinf(Z[ii,jj])
            
            if xGood and yGood and zGood:
                x = x + [ X[ii,jj] ]
                y = y + [ Y[ii,jj] ]
                z = z + [ Z[ii,jj] ]
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    return [x,y,z]


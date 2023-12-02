# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:39:02 2022

@author: dirge
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:56:52 2020
@author: Kevin Gomez, Nathan Aviles
Masel Lab
see Bertram, Gomez, Masel 2016 for details of Markov chain approximation
see Bertram & Masel 2019 for details of lottery model
"""

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



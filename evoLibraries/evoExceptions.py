# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:12:10 2025

@author: Owner
"""

#%% *****************************************************************************
#                       Evo Eceptions Class
# *****************************************************************************
        
class EvoExceptions(Exception):
    """Base Class for Evo Libary Exceptions"""
    
# --------------------------------------------------------------------------

class EvoInvalidInput(EvoExceptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.custom_kwarg = kwargs.get('custom_kwarg')

# --------------------------------------------------------------------------    
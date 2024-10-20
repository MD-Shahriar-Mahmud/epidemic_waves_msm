# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:54:03 2024

@author: shahriar
"""

def format_label(param_name):
    return r'${}$'.format(param_name) if param_name in {'c', 'k','R_0'} else r'$\{}$'.format(param_name)

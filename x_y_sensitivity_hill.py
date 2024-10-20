# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:23:58 2024

@author: prism
"""

import numpy as np
from funcs.simulation_hill_func import simulate
from funcs.find_peak_nums import find_peak_nums

def x_y_sensitivity(x_range, y_range, x_param_name, y_param_name):
    matrix = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        print(f'Now progressing with {y_param_name} = {y} ; iteration = {i}/200')
        for j, x in enumerate(x_range):
            parameters = {x_param_name: x, y_param_name: y, 'dt': 1}
            ts, results, _, _ = simulate(**parameters)
            wave_num = find_peak_nums(results)
            matrix[i, j] = wave_num
    return matrix
    
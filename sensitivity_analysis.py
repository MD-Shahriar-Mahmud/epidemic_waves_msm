# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:35:20 2024

@author: shahriar
"""

import numpy as np
from func_x_y_sensitivity_hill import x_y_sensitivity
from func_file_utils import save_data, load_data
from func_draw_heatmap import draw_heatmap

def calculate_sensitivity_matrices_for_hill():
    tau_range = np.linspace(0, 20, 21)
    c_range = np.linspace(1, 17, 21)
    k_range = np.linspace(4, 64, 21)
    beta_range = np.linspace(0.2, 1, 21)
    gamma_range = np.linspace(0.2, 0.4, 21)    
    
    matrix_tau_c = x_y_sensitivity(tau_range, c_range, 'tau', 'c')
    matrix_tau_k = x_y_sensitivity(tau_range, k_range, 'tau', 'k')
    matrix_tau_beta = x_y_sensitivity(tau_range, beta_range, 'tau', 'beta')
    matrix_tau_gamma = x_y_sensitivity(tau_range, gamma_range, 'tau', 'gamma')
    
    data_dict = {
        'beta_range': beta_range,
        'gamma_range': gamma_range,
        'c_range': c_range,
        'k_range': k_range,
        'tau_range': tau_range,
        'matrix_tau_c': matrix_tau_c,
        'matrix_tau_k': matrix_tau_k,
        'matrix_tau_beta': matrix_tau_beta,
        'matrix_tau_gamma': matrix_tau_gamma,
    }
    
    save_data(data_dict,'sensitivity_matrices_for_hill_with_'+str(len(tau_range))+'_mesh.pkl')


if __name__ == "__main__":
    
    # calculate_sensitivity_matrices_for_hill()
    
    try:
        data = load_data('sensitivity_matrices_for_hill_with_21_mesh.pkl')
        print("Data loaded successfully")
    except FileNotFoundError as e:
        print(e)
    
    draw_heatmap(data['matrix_tau_c'], data['tau_range'], data['c_range'], 'tau', 'c', 'hill_figs')
    draw_heatmap(data['matrix_tau_k'], data['tau_range'], data['k_range'], 'tau', 'k', 'hill_figs')
    draw_heatmap(data['matrix_tau_beta'], data['tau_range'], data['beta_range'], 'tau', 'beta', 'hill_figs')
    draw_heatmap(data['matrix_tau_gamma'], data['tau_range'], data['gamma_range'], 'tau', 'gamma', 'hill_figs')
    
    
    
    
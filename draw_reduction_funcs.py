# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:05:54 2024

@author: shahriar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter    
 
def response_func_graph(c_vals, k_hill_vals, k_sigmoid_vals):
    
    folder_name = os.path.join('data')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    total_population = 5000
    I = np.logspace(1, np.log10(total_population), 10000)
    # colors = ['r','b','g','C1']
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    for c in np.array(c_vals, dtype=int):
        c_scaled = int(c / 50)
        ax.axvline(x=c_scaled, color='gray', linestyle='--')
        count = 0
        for k_hill in k_hill_vals:
            r_hill = (np.log10(I))**k_hill / ((np.log10(I))**k_hill + (np.log10(c))**k_hill)
            ax.semilogx(I / 50, r_hill, #color=colors[count], 
                        linestyle=':',label=f'hill: $k_h$ = {k_hill}',linewidth=3)
            count += 1
        for k_sigmoid in k_sigmoid_vals:
            r_sigmoid = 1 / (1 + np.exp(-k_sigmoid * (I - c)))
            ax.semilogx(I / 50, r_sigmoid, #color=colors[count], 
                        linestyle='-',label=f'sigmoid: $k_s$ = {k_sigmoid}')
            count += 1
            
    ax.set_xlabel('prevalence (in % of total population)')
    ax.set_ylabel('contact reduction')
        
    if c == 50:
        ax.set_xlim([0.2,5])
        ax.set_xticks([0.2,1,5])
    elif c==100:
        ax.set_xlim([0.4,10])
        ax.set_xticks([0.4,2,10])
    elif c==200:
        ax.set_xlim([1,15])
        ax.set_xticks([1,4,15])
    
    ax.grid(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    ax.spines[['right','top']].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{str(x)}%'))
    file_path = os.path.join(folder_name, 'response_func_graphs.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    
if __name__ == "__main__":
    
    c_vals = [100]
    kh_vals = [[16, 24]]
    ks_vals = [[0.1]]
    
    for c, ks, kh in zip(c_vals, ks_vals, kh_vals):
        response_func_graph(c_vals=[c], k_hill_vals=kh, k_sigmoid_vals=ks)
        
        
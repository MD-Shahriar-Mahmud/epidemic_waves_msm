# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:36:27 2024

@author: shahriar
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def infer_index_given_min_max_number(value,min_value,max_value,number):
    dx = (number-1)/(max_value-min_value)
    return (value-min_value) * dx


def infer_ticks(ticks,parameter_values):
    min_value = min(parameter_values)
    max_value = max(parameter_values)
    number = len(parameter_values)
    return np.array([infer_index_given_min_max_number(el,min_value,max_value,number) for el in ticks])


def draw_heatmap(matrix,x_range,y_range,x_param_name,y_param_name,subfolder_name,global_max=None,global_min=None):
    folder_name = os.path.join('data', subfolder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    local_min = global_min if global_min is not None else np.array(matrix).min()
    local_max = global_max if global_max is not None else np.array(matrix).max()
    n_bins = int(local_max - local_min + 1)
    cmap = plt.get_cmap('jet', n_bins)
    bounds = np.linspace(local_min, local_max + 1, n_bins + 1)
    norm = BoundaryNorm(bounds, cmap.N)
        
    df = pd.DataFrame(matrix, index=np.round(y_range, 2), columns=np.round(x_range, 2))
    
    # plt.figure(figsize=(5, 5))
    plt.figure(figsize=(6, 4.5))
    ax = sns.heatmap(df, cmap=ListedColormap(cmap(np.linspace(0, 1, n_bins))), 
                     norm=norm, annot=False, cbar_kws={'label': 'Number of Waves'})
    
    cbar = ax.collections[0].colorbar
    exten = 2 if (local_max-local_min)%2==0 else 1
    if local_max > 20:
        tick_locs = np.arange(local_min + 0.5, local_max + exten + 0.5)
        # print(tick_locs)
        cbar.set_ticks(tick_locs[(tick_locs-0.5) % 2 == 0] - 1)
        tick_labs = np.arange(local_min, local_max + exten, dtype=int)
        # print(tick_labs)
        cbar.set_ticklabels(tick_labs[tick_labs % 2 == 0] - 1)
        cbar.ax.tick_params()
        cbar.set_label('number of waves')
    else:            
        tick_locs = np.arange(local_min + 0.5, local_max + 1.5)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(np.arange(local_min, local_max + 1, dtype=int))
        cbar.ax.tick_params()
        cbar.set_label('number of waves')
      
    ax.tick_params(axis='both', which='major')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.invert_yaxis()
    
    a = y_range.min()
    b = y_range.max()
    y_axis_ticks_pos = np.linspace(a, b, 5) + (b-a)/((len(y_range)-1)*2)
    y_axis_ticks_vals = np.linspace(a, b, 5)
    if y_param_name == 'k':
        y_axis_ticks_vals = np.array([f"{val:.0f}" for val in y_axis_ticks_vals])
        ax.set_ylabel("sensitivity of response ($k_h$)")
        h_text = 'high'
        l_text = 'low'
    elif y_param_name == 'c':
        y_axis_ticks_vals = np.array([f"{val:.1f}" for val in y_axis_ticks_vals])
        ax.set_ylabel("half-maximal reduction point ($c$)\nin % of total population")
        h_text = 'high'
        l_text = 'low'
    elif y_param_name == 'beta':
        y_axis_ticks_vals = np.array([f"{val:.2f}" for val in y_axis_ticks_vals])
        ax.set_ylabel(r"transmission rate ($\beta$)")
        h_text = 'high'
        l_text = 'low'
    elif y_param_name == 'gamma':
        y_axis_ticks_vals = np.array([f"{val:.2f}" for val in y_axis_ticks_vals])
        ax.set_ylabel(r"recovery rate ($\gamma$)")
        h_text = 'high'
        l_text = 'low'
    else:
        y_axis_ticks_vals = np.array([f"{val:.2f}" for val in y_axis_ticks_vals])
        h_text = 'high'
        l_text = 'low'
        
    y_axis_vals = np.linspace(a,b,len(y_range))
    ax.set_yticks(infer_ticks(y_axis_ticks_pos,y_axis_vals))
    ax.set_yticklabels(list(map(str,y_axis_ticks_vals)))
    
    aa = x_range.min()
    bb = x_range.max()
    x_axis_ticks_pos = np.linspace(aa, bb, 5) + (bb-aa)/((len(x_range)-1)*2)
    x_axis_ticks_vals = np.linspace(aa, bb, 5)
    x_axis_ticks_vals = np.array([f"{val:.0f}" for val in x_axis_ticks_vals])
    x_axis_vals = np.linspace(aa,bb,len(x_range))
    ax.set_xticks(infer_ticks(x_axis_ticks_pos,x_axis_vals))
    ax.set_xticklabels(list(map(str,x_axis_ticks_vals)))
    ax.set_xlabel(r"delay time ($\tau$)")
    
    y_min, y_max = ax.get_ylim()    
    ax.text(x=-(len(x_range)/5), y=y_max - 0.5, s=h_text, ha='center', va='center')
    ax.text(x=-(len(x_range)/5), y=y_min + 0.5, s=l_text, ha='center', va='center')

    
    file_path = os.path.join(folder_name, 'sen_'+str(x_param_name)+'_'+str(y_param_name)+'_with_'+str(len(x_range))+'_mesh.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    
     
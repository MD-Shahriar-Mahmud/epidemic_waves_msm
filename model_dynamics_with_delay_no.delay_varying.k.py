# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:32:40 2024

@author: shahriar
"""

import os
import numpy as np
from func_simulation_hill_func import simulate
import matplotlib.pyplot as plt

def no_delay_dynamics():
    fig, ax = plt.subplots(figsize=(5,3))
    ax2 = ax.twinx()
    
    ts, results, reduction, Reff = simulate(case=0)
    I = results[:, 1]
    ax.plot(ts, I, color=color[0], linestyle='-')
    ax2.plot(ts, Reff, color=color[0], linestyle='--')
    
    ts, results, reduction, Reff = simulate(case=1)
    I = results[:, 1]
    ax.plot(ts, I, color=color[1], linestyle='-')
    ax2.plot(ts, Reff, color=color[1], linestyle='--')
        
    ax.set_xlabel('time in days')
    ax.set_ylabel('prevalence\n(in % of total population)')
    ax2.set_ylabel('effective reproduction\nnumber ($R_{eff}$)')
    ax.set_xlim([0,300])
    ax.set_ylim(bottom=None, top=20)
    ax2.set_ylim([-0.05*Reff.max(),1.1*Reff.max()])
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', linestyle='-', lw=2, label='prevalence'),
                        Line2D([0], [0], color='k', linestyle='--', lw=2, label='$R_{eff}$'),
                        Line2D([0], [0], color=color[0], linestyle='-', lw=7, label='no contact reduction'),
                        Line2D([0], [0], color=color[1], linestyle='-', lw=7, label='hill as contact reduction')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    ax.spines[['right','top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    

def delay_dynamics(delay_range=[5]):
    for delay in delay_range:
        ts, results, reduction, Reff = simulate(tau=delay)
        I = results[:, 1]
        fig, ax = plt.subplots(figsize=(5,3))
        ax2 = ax.twinx()
        
        ax.plot(ts, I, color=color[0], linestyle='-')
        ax.set_xlabel('time in days (delay = 5 days)')
        ax.set_ylabel('prevalence\n(in % of total population)', color=color[0])
        ax.set_xlim([0,140])
        ax.set_ylim(bottom=None, top=4)
        ax.spines[['top']].set_visible(False)
        
        ax2.plot(ts, Reff, color=color[1], linestyle='--')
        ax2.plot(ts,np.ones(len(ts)), linestyle=':', color='gray', linewidth=1)
        ax2.set_ylabel('effective reproduction number', color=color[1])
        ax2.set_ylim([-0.05*Reff.max(),1.1*Reff.max()])
        ax2.spines[['top']].set_visible(False)
        
        file_path = os.path.join(folder_name, 'delay_dynamics_with_delay_of_'+str(delay)+'_days.pdf')
        plt.savefig(file_path, format='pdf', bbox_inches='tight')
        plt.show()


def dynamics_for_varying_k(delay=5, k_range=[8, 16, 24]):
    I = []
    r = []
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, k in enumerate(k_range):
        ts, results, reduction, Reff = simulate(k=k, tau=delay)
        I.append(results[:, 1])
        r.append(reduction)
        ax.plot(ts, I[i], color=color[i], label=r'$%g$' % (int(k)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_ylabel('prevalence \n(in % of total population)')
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim(bottom=None, top=4.2)
    legend = ax.legend(loc='best', frameon=False)
    legend.set_title("sensitivity to\nresponse ($k_h$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'dynamics_for_varying_k.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, k in enumerate(k_range):
        ax.plot(ts, r[i], color=color[i], label=r'$%g$' % (int(k)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_ylabel('contact reduction')
    ax.grid(False)
    ax.set_xlim([0, 150])
    legend = ax.legend(loc='best', frameon=False)
    legend.set_title("sensitivity to\nresponse ($k_h$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'reduction_for_varying_k.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    
    folder_name = os.path.join('data', 'hill_figs')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    color = ['k','r','b']
    
    no_delay_dynamics()
    delay_dynamics(delay_range=[5])
    dynamics_for_varying_k(delay=5, k_range=[8, 16, 24])
    
    
    
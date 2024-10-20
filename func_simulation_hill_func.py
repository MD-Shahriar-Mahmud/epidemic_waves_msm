# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:41:23 2024

@author: shahriar
"""

import numpy as np

def SIR_with_delayed_reduction(y, t, beta, gamma, c, k, N, tau, history, current_step, dt, case=1):
    S, I, R = y
    if tau > 0:
        idx = max(current_step - int(tau / dt), 0)
        delayed_I = history[idx]
    else:
        delayed_I = I
    
    epsilon = 1e-10
    log_I = np.log10(np.maximum(delayed_I, epsilon))
    log_c = np.log10(c)
    log_I_safe = np.log(np.maximum(log_I, epsilon))
    log_c_safe = np.log(np.maximum(log_c, epsilon))
    
    max_exponent = 700
    exponent = np.clip(k * (log_c_safe - log_I_safe), -max_exponent, max_exponent)
    hill_reduction = case / (1 + np.exp(exponent))
    
    dS = -beta * (1 - hill_reduction) * S * I / N
    dI = beta * (1 - hill_reduction) * S * I / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])

def rk4(y, t, dt, model, beta, gamma, c, k, N, tau, history, current_step, case=1):
    k1 = dt * model(y, t, beta, gamma, c, k, N, tau, history, current_step, dt, case)
    k2 = dt * model(y + 0.5 * k1, t + 0.5 * dt, beta, gamma, c, k, N, tau, history, current_step, dt, case)
    k3 = dt * model(y + 0.5 * k2, t + 0.5 * dt, beta, gamma, c, k, N, tau, history, current_step, dt, case)
    k4 = dt * model(y + k3, t + dt, beta, gamma, c, k, N, tau, history, current_step, dt, case)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4_solver(ts, x0, beta, gamma, tau, c, k, N, dt, case=1):
    steps = len(ts)
    history = np.zeros(steps)
    delayed_I = np.zeros(steps)
    history[:int(tau / dt)] = x0[1]
    results = np.zeros((steps, 3))
    results[0] = x0
    for i in range(1, steps):
        results[i] = rk4(results[i-1], ts[i-1], dt, SIR_with_delayed_reduction, beta, gamma, c, k, N, tau, history, i, case)
        history[i] = results[i, 1]
        delayed_I[i] = history[max(i - int(tau / dt), 0)]
    return results, delayed_I

def simulate(N=5000, I_ini=1, beta=0.4, gamma=0.2, c=2, k=16, tau=0, dt=0.1, case=1):
    I0 = 1
    t_end = 500
    ts = np.linspace(0, t_end, int(t_end / dt) + 1)
    x0 = (N - I0, I0, 0)
    c = c*N/100
    results, delayed_I = rk4_solver(ts, x0, beta, gamma, tau, c, k, N, dt, case)
    
    epsilon = 1e-10
    log_I = np.log10(np.maximum(delayed_I, epsilon))
    log_c = np.log10(c)
    log_I_safe = np.log(np.maximum(log_I, epsilon))
    log_c_safe = np.log(np.maximum(log_c, epsilon))
    
    max_exponent = 700
    exponent = np.clip(k * (log_c_safe - log_I_safe), -max_exponent, max_exponent)
    reduction = case / (1 + np.exp(exponent))

    Reff = (1 - reduction) * (results[:, 0] / N) * (beta / gamma)
    return ts, results/N*100, reduction, Reff

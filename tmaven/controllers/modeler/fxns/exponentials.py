import numpy as np
import numba as nb


def single_exp_surv(tau, k, A):
    return A*np.exp(-k*tau)

def double_exp_surv(tau, k1, k2,A,B):
    return A*np.exp(-k1*tau) + B*np.exp(-k2*tau)

def triple_exp_surv(tau, k1, k2, k3, A, B, C):
    return A*np.exp(-k1*tau) + B*np.exp(-k2*tau) + C*np.exp(-k3*tau)

def stretched_exp_surv(tau, k, beta, A):
    return A*np.exp(-(k*tau)**beta)

def single_exp_hist(tau, k, A):
    return A*k*np.exp(-k*tau)

def double_exp_hist(tau, k1, k2, A, B):
    return A*k1*np.exp(-k1*tau) + B*k2*np.exp(-k2*tau)

def triple_exp_hist(tau, k1, k2, k3, A, B, C):
    return A*k1*np.exp(-k1*tau) + B*k2*np.exp(-k2*tau) + C*k3*np.exp(-k3*tau)

def stretched_exp_hist(tau, k, beta, A):
    return A*np.exp(-(k*tau)**beta)*((tau)**beta)*beta*(k**(beta - 1))
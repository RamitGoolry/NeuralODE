# Runge-Kutta 4th order ODE solver

import torch
from torch import Tensor

import numpy as np

def euler(z0, f, t0, t1, dt=1e-3):
    '''
    Euler ODE Solver

    Parameters:
    z0: initial condition
    f: function to solve => f(t, z)
    t0: initial time
    t1: final time
    dt: time  

    Returns:
    solution at time t1
    '''

    t = t0
    z = z0

    while t < t1:
        z += f(t, z) * dt
        t += dt

    return z

def runge_kutta4(z0, f, t0, t1, dt=1e-3):
    '''
    Runge-Kutta 4 ODE Solver

    Parameters:
    z0: initial condition
    f: function to solve => f(z, t)
    t0: initial time
    t1: final time
    dt: time step

    Returns:
    solution at time t1
    '''

    t = t0
    z = z0

    while t < t1:
        k1 = f(z, t)
        k2 = f(z + k1 * dt / 2, t + dt / 2)
        k3 = f(z + k2 * dt / 2, t + dt / 2)
        k4 = f(z + k3 * dt, t + dt)

        z += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        t += dt

    return z
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#global parameters
G = 1
m1 = 10**-6
m2 = 10**-8
phi0 = 4 #the location of the peak of the additional gaussian potential
mu = 0.5

def V(phi,chi):
    return 1./2*m1**2*phi**2*(1 + np.tanh(m2**chi)*np.exp(-(phi - phi0)**2/(2*mu)))

def V_phi(phi,chi):
    eps = 10**-5
    return (V(phi + eps,chi) - V(phi,chi))/eps

def V_chi(phi,chi):
    eps = 10**-5
    return (V(phi,chi+eps) - V(phi,chi))/eps

def H(phi,psi,chi,xi):
    return np.sqrt(2*V(phi,chi)/(3/(4*np.pi*G) - (psi**2 + xi**2)))

def H_phi(phi,psi,chi,xi):
    eps = 10**-5
    return (H(phi+eps,psi,chi,xi) - H(phi,psi,chi,xi))/eps

def H_psi(phi,psi,chi,xi):
    eps = 10**-5
    return (H(phi,psi+eps,chi,xi) - H(phi,psi,chi,xi))/eps

def H_chi(phi,psi,chi,xi):
    eps = 10**-5
    return (H(phi,psi,chi+eps,xi) - H(phi,psi,chi,xi))/eps


def H_xi(phi,psi,chi,xi):
    eps = 10**-5
    return (H(phi,psi,chi,xi+eps) - H(phi,psi,chi,xi))/eps



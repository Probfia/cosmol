#calculate the relation between the initial values of phi and the numbers of e-fold inflation

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#global parameters
G = 1
m = 10**-6

#Hubble parameter and its partial derivs
def H(phi,psi):
    return np.sqrt(8*np.pi*G/3*(1./2*m**2*phi**2))/np.sqrt(1-1./2*psi**2)

def dHdphi(phi,psi):
    eps = 10**-5
    return (H(phi+eps,psi) - H(phi,psi))/eps

def dHdpsi(phi,psi):
    eps = 10**-5
    return (H(phi,psi+eps)-H(phi,psi))/eps

#oprator d/du acting on phi, psi
def d_dlnus(var,lna,params):
    phi, psi = var
    H0, m = params
    derivs = [psi,
              -1./(1 + 1/H(phi,psi)*dHdpsi(phi,psi)*psi)
              *((1/H(phi,psi)*dHdphi(phi,psi)*psi+3)*psi + m**2/H(phi,psi)**2*phi)]
    return derivs

#initial conditions
phi0 = 17
psi0 = 0
H0 = H(phi0,psi0)
params = [H0,m]
var0 = [phi0,psi0]

lnaStop = 2000
num_points = 10000
lna = np.linspace(0,lnaStop,num_points)

#return the solution of the ode, conp. 0 is the array of phis and conp. 1 for psis
sol = odeint(d_dlnus,var0,lna,args=(params,))

phi = sol[:,0]
eps = 1./(4*np.pi*phi**2)
arg_crit = np.argmax(eps>1)
N = lna[arg_crit]

#calculate spectral power n
n = 1 - 4*eps
n_60 = np.interp(N-60,lna,n) #return the value of lna where n(lna) = N - 60
print(n_60)

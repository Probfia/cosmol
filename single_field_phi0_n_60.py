#calculate the relation between the initial values of phi and the numbers of e-fold inflation

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

G = 1
m = 10**-6

def H(phi,psi):
    return np.sqrt(8*np.pi*G/3*(1./2*m**2*phi**2))/np.sqrt(1-1./2*psi**2)

def dHdphi(phi,psi):
    eps = 10**-5
    return (H(phi+eps,psi) - H(phi,psi))/eps

def dHdpsi(phi,psi):
    eps = 10**-5
    return (H(phi,psi+eps)-H(phi,psi))/eps

def d_dlnus(var,lna,params):
    phi, psi = var
    H0, m = params
    derivs = [psi,
              -1./(1 + 1/H(phi,psi)*dHdpsi(phi,psi)*psi)
              *((1/H(phi,psi)*dHdphi(phi,psi)*psi+3)*psi + m**2/H(phi,psi)**2*phi)]
    return derivs

phi_inti_vals = [0.02*x for x in range(100,300)]
N_vals = []
n_60_vals = []
for phi0 in phi_inti_vals:
    #phi0 = 17
    psi0 = 0
    H0 = H(phi0,psi0)
    params = [H0,m]
    var0 = [phi0,psi0]

    lnaStop = 2000
    num_points = 10000
    lna = np.linspace(0,lnaStop,num_points)

    sol = odeint(d_dlnus,var0,lna,args=(params,))

    phi = sol[:,0]
    eps = 1./(4*np.pi*phi**2)
    arg_crit = np.argmax(eps>1)
    n = 1 - 4*eps

    N = lna[arg_crit]
    N_vals.append(N)
    n_60 = np.interp(N-60,lna,n)
    n_60_vals.append(n_60)

plt.plot(phi_inti_vals,n_60_vals)
plt.show()

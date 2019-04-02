import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#global parameters
G = 1
m1 = 10**-6
m2 = 10**-3
phipk = 4 #the location of the peak of the additional gaussian potential
mu = 1
alpha = 100

def V(phi,chi):
    return 1./2*m1**2*phi**2*(1 + alpha*np.tanh(m2*chi)*np.exp(-(phi - phipk)**2/(2*mu)))

def V_phi(phi,chi):
    eps = 10**-6
    return (V(phi+eps/2,chi) - V(phi-eps/2,chi))/eps

def V_chi(phi,chi):
    eps = 10**-6
    return (V(phi,chi+eps/2) - V(phi,chi-eps/2))/eps

def d_dlna(var,lna,paras):
    H,phi,psi,chi,xi = var
    m1,m2 = paras
    return [(8*np.pi*G/(3*H))*(V(phi,chi) - H**2*(psi**2 + xi**2)) - H,
            psi,
            -2*psi - V_phi(phi,chi)/H**2 - (8*np.pi*G*psi/3)*(V(phi,chi)/H**2 - psi**2 - xi**2),
            xi,
            -2*xi - V_chi(phi,chi)/H**2 - (8*np.pi*G*xi/3)*(V(phi,chi)/H**2 - psi**2 - xi**2)]

def thirdH(phi,psi,chi,xi):
    return np.sqrt(2*V(phi,chi)/(3/(4*np.pi*G) - (psi**2 + xi**2)))

phi0 = 6
psi0 = 0
chi0 = 1
xi0 = 0
H0 = thirdH(phi0,psi0,chi0,xi0)
paras = [m1,m2]
var0 = [H0,phi0,psi0,chi0,xi0]

lnaStop = 200
num_of_pts = 100000
lnavals = np.linspace(0,lnaStop,num_of_pts)

sol = odeint(d_dlna,var0,lnavals,args=(paras,))
phivals = sol[:,1]
psivals = sol[:,2]
chivals = sol[:,3]
xivals = sol[:,4]

plt.plot(lnavals,phivals)
plt.plot(lnavals,chivals)

plt.show()
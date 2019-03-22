import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#global parameters
G = 1
m1 = 10**-6
m2 = 10**-8
phipk = 4 #the location of the peak of the additional gaussian potential
mu = 1
alpha = 1000000000

def V(phi,chi):
    return 1./2*m1**2*phi**2*(1 + alpha*np.exp(m2*chi)*np.exp(-(phi - phipk)**2/(2*mu)))

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

def d_du(var,lna,paras):
    phi, psi, chi, xi = var
    H0, m1, m2 = paras
    result = [psi,    # dphi/du
              -(H_xi(phi,psi,chi,xi)*(V_phi(phi,chi)*xi - V_chi(phi,chi)*psi)
                + H(phi,psi,chi,xi)**2*(H_phi(phi,psi,chi,xi)*psi**2 + H_xi(phi,psi,chi,xi)*psi*xi
                + 3*H(phi,psi,chi,xi)*psi) + H(phi,psi,chi,xi)*V_phi(phi,chi))/
              (H(phi,psi,chi,xi)**2*(H_xi(phi,psi,chi,xi)*xi + H_psi(phi,psi,chi,xi)*psi + H(phi,psi,chi,xi))),
              xi,     # dchi/du
              -(H_psi(phi,psi,chi,xi)*(V_chi(phi,chi)*psi - V_phi(phi,chi)*xi)
                + H(phi,psi,chi,xi)**2*(H_chi(phi,psi,chi,xi)*xi**2 + H_phi(phi,psi,chi,xi)*psi*xi +
                                       3*H(phi,psi,chi,xi)*xi) + H(phi,psi,chi,xi)*V_chi(phi,chi))/
              (H(phi,psi,chi,xi)**2*(H_xi(phi,psi,chi,xi)*xi + H_psi(phi,psi,chi,xi)*psi + H(phi,psi,chi,xi)))]
    return result

phi0 = 10
psi0 = 0
chi0 = 2
xi0 = -0.005
H0 = H(phi0,psi0,chi0,xi0)
paras = [H0,m1,m2]
var0 = [phi0,psi0,chi0,xi0]

lnaStop = 2000
num_of_pts = 10000
lna = np.linspace(0,lnaStop,num_of_pts)

sol = odeint(d_du,var0,lna,args=(paras,))
phi = sol[:,0]
chi = sol[:,2]

plt.plot(lna,phi)
plt.plot(lna,chi)
plt.show()

with open('dfsolution1.txt','w') as fobj:
    for i in range(len(lna)):
        fobj.write(str(lna[i]) + ' ' + str(phi[i]) + ' ' + str(chi[i]) + ' \n')

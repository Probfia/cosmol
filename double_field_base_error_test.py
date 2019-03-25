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
xi0 = 0
H0 = H(phi0,psi0,chi0,xi0)
paras = [H0,m1,m2]
var0 = [phi0,psi0,chi0,xi0]

lnaStop = 2000
num_of_pts = 1000000
lna = np.linspace(0,lnaStop,num_of_pts)

sol = odeint(d_du,var0,lna,args=(paras,))
phi = sol[:,0]
psi = sol[:,1]
chi = sol[:,2]
xi = sol[:,3]

def first_deriv(yvals,xvals):
    result = np.zeros(len(yvals) - 1)
    for k in range(len(yvals) - 1):
        dy = yvals[k+1] - yvals[k]
        dx = xvals[k+1] - xvals[k]
        result[k] = dy/dx
    return result

def second_deriv(yvals,xvals):
    temp = first_deriv(yvals,xvals)
    return first_deriv(temp,xvals)

Hvals = H(phi,psi,chi,xi)
dH_du = first_deriv(Hvals,lna)
dphi_du = first_deriv(phi,lna)
ddphi_du2 = second_deriv(phi,lna)

l = min(len(Hvals),len(dH_du),len(dphi_du),len(ddphi_du2),len(phi),len(chi))

Hvals = Hvals[0:l]
dH_du = dH_du[0:l]
dphi_du = dphi_du[0:l]
ddphi_du2 = ddphi_du2[0:l]
phi = phi[0:l]
chi = chi[0:l]
psi = psi[0:l]
xi = xi[0:l]

err = Hvals**2*(ddphi_du2 + 3*dphi_du) + Hvals*dphi_du*dH_du + V_phi(phi,chi)

plt.plot(lna[0:len(err)],err)

plt.show()
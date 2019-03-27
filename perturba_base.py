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

def V_phi_phi(phi,chi):
    eps = 10**-5
    return (V_phi(phi+eps,chi) - V_phi(phi,chi))/eps

def V_phi_chi(phi,chi):
    eps = 10**-5
    return (V_phi(phi,chi+eps)-V_phi(phi,chi))/eps

def V_chi_chi(phi,chi):
    eps = 10**-5
    return (V_chi(phi,chi+eps) - V_chi(phi,chi))/eps

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

# d/dlna action on bg fields phi and chi
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
num_of_pts = 100000

lnavals = np.linspace(0,lnaStop,num_of_pts)

sol = odeint(d_du,var0,lnavals,args=(paras,))

phivals = sol[:,0]
psivals = sol[:,1]
chivals = sol[:,2]
xivals = sol[:,3]

Hvals = H(phivals,psivals,chivals,xivals)
epsvals = 1./2 * (psivals**2 + xivals**2)


c_11 = V_phi_phi(phivals,chivals)/Hvals**2 + 1./Hvals**2*(2*psivals*V_phi(phivals,chivals)) + (3 - epsvals)*psivals**2

c_12 = V_phi_chi(phivals,chivals)/Hvals**2 + 1./Hvals**2*(psivals*V_chi(phivals,chivals)
                                                          + xivals*V_phi(phivals,chivals))+ (3 - epsvals)*psivals*xivals

c_21 = c_12

c_22 = V_chi_chi(phivals,chivals)/Hvals**2 + 1./Hvals**2*(2*xivals*V_chi(phivals,chivals)) + (3 - epsvals)*xivals**2


dotavals = np.exp(lnavals)*Hvals
k = 0.002

# get_val_for_given_lna, return arr(lna)
def get_val(arr,x):
    return np.interp(x,lnavals,arr)

# d/dlna action on Psi_{11}  Theta = dPsi/dlna , see eq.(3.14) on multi_field_codes
def d_dNe_ij(var,lna,paras):
    Psi_11, Theta_11, Psi_22, Theta_22 = var
    H0, m1, m2 = paras
    result = [Theta_11,
              -((1-get_val(epsvals,lna))*Theta_11 + (k**2/get_val(dotavals,lna)**2 - 2 + get_val(epsvals,lna))*Psi_11 + get_val(c_11,lna)*Psi_11),
              Theta_22,
              -((1-get_val(epsvals,lna))*Theta_22 + (k**2/get_val(dotavals,lna)**2 - 2 + get_val(epsvals,lna))*Psi_22 + get_val(c_22,lna)*Psi_22)]
    return result


# initial values of Psi_{ij}, dPsi/dlna, only i=j have non-zero values
Psi0_22 = Psi0_11 = 1./np.sqrt(2*k)
Theta0_22 = Theta0_11 = -1./dotavals[0]*np.sqrt(k/2)
ptbvar0 = [Psi0_11,Theta0_11,Psi0_22,Theta0_22]

ptb_sol = odeint(d_dNe_ij,ptbvar0,lnavals,args=(paras,))

Psi_11_vals = ptb_sol[:,0]
Psi_22_vals = ptb_sol[:,2]

plt.plot(lnavals,Psi_11_vals)
plt.plot(lnavals,Psi_22_vals)

plt.show()
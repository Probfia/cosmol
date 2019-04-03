import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#global parameters
Mpl = 1
G = 1./(8*np.pi*Mpl**2)
m1 = 10**-6*Mpl
m2 = 10**-1*Mpl
phipk = 8*Mpl #the location of the peak of the additional gaussian potential
mu = 4*Mpl^2 # sigma^2 of the additional gaussian potential
alpha = 5

def V(phi,chi):
    return 1./2*m1**2*phi**2*(1 + alpha*np.tanh(m2*chi/Mpl**2)*np.exp(-(phi - phipk)**2/(2*mu)))

def V_phi(phi,chi):
    eps = 10**-6*Mpl
    return (V(phi+eps/2,chi) - V(phi-eps/2,chi))/eps

def V_chi(phi,chi):
    eps = 10**-6*Mpl
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

phi0 = 15
psi0 = 0
chi0 = 2
xi0 = 0
H0 = thirdH(phi0,psi0,chi0,xi0)
paras = [m1,m2]
var0 = [H0,phi0,psi0,chi0,xi0]

lnaStop = 40
num_of_pts = 10000
lnavals = np.linspace(0,lnaStop,num_of_pts)

sol = odeint(d_dlna,var0,lnavals,args=(paras,))
Hvals = sol[:,0]
phivals = sol[:,1]
psivals = sol[:,2]
chivals = sol[:,3]
xivals = sol[:,4]



plt.plot(lnavals,Hvals)
plt.show()

# perturbation fields
epsvals = 0.5*(psivals**2 + xivals**2)

def V_phi_phi(phi,chi):
    eps = 10**-6*Mpl
    return (V_phi(phi+eps/2,chi) - V_phi(phi-eps/2,chi))/eps

def V_phi_chi(phi,chi):
    eps = 10**-6*Mpl
    return (V_phi(phi,chi+eps/2) - V_phi(phi,chi-eps/2))/eps

def V_chi_chi(phi,chi):
    eps = 10**-6*Mpl
    return (V_chi(phi,chi+eps/2) - V_chi(phi,chi-eps/2))/eps

c_11 = V_phi_phi(phivals,chivals)/Hvals**2 + 1./Hvals**2*(2*psivals*V_phi(phivals,chivals)) + (3 - epsvals)*psivals**2

c_12 = V_phi_chi(phivals,chivals)/Hvals**2 + 1./Hvals**2*(psivals*V_chi(phivals,chivals)
                                                          + xivals*V_phi(phivals,chivals))+ (3 - epsvals)*psivals*xivals

c_21 = c_12

c_22 = V_chi_chi(phivals,chivals)/Hvals**2 + 1./Hvals**2*(2*xivals*V_chi(phivals,chivals)) + (3 - epsvals)*xivals**2

dotavals = np.exp(lnavals)*Hvals
k = 20*H0

# get_val_for_given_lna, return arr(lna)
def get_val(arr,x):
    return np.interp(x,lnavals,arr)

def d_dlna_ij(var,lna,paras):
    RePsi_11, ReTheta_11, ImPsi_11, ImTheta_11, RePsi_22, ReTheta_22, ImPsi_22, ImTheta_22 = var
    m1, m2 = paras
    result = [ReTheta_11,
              -((1-get_val(epsvals,lna))*ReTheta_11 + (k**2/get_val(dotavals,lna)**2 - 2 + get_val(epsvals,lna))*RePsi_11 + get_val(c_11,lna)*RePsi_11),
              ImTheta_11,
              -((1-get_val(epsvals,lna))*ImTheta_11 + (k**2/get_val(dotavals,lna)**2 - 2 + get_val(epsvals,lna))*ImPsi_11 + get_val(c_11,lna)*ImPsi_11),
              ReTheta_22,
              -((1-get_val(epsvals,lna))*ReTheta_22 + (k**2/get_val(dotavals,lna)**2 - 2 + get_val(epsvals,lna))*RePsi_22 + get_val(c_22,lna)*RePsi_22),
              ImTheta_22,
              -((1-get_val(epsvals,lna))*ImTheta_22 + (k**2/get_val(dotavals,lna)**2 - 2 + get_val(epsvals,lna))*ImPsi_22 + get_val(c_22,lna)*ImPsi_22)]
    return result

RePsi_22_0 = RePsi_11_0 = (1./np.sqrt(2*k))
ImPsi_22_0 = ImPsi_11_0 = 0
ReTheta_22_0 = ReTheta_11_0 = 0
ImTheta_22_0 = ImTheta_11_0 = -(1./dotavals[0])*np.sqrt(k/2)

ptb_init = [RePsi_11_0, ReTheta_11_0, ImPsi_11_0, ImTheta_11_0, RePsi_22_0, ReTheta_22_0, ImPsi_22_0, ImTheta_22_0]

ptb_sol = odeint(d_dlna_ij,ptb_init,lnavals,args=(paras,))

RePsi_11_vals = ptb_sol[:,0]
ImPsi_11_vals = ptb_sol[:,1]

plt.plot(lnavals,RePsi_11_vals/np.exp(lnavals))
plt.plot(lnavals,ImPsi_11_vals/np.exp(lnavals))
plt.show()










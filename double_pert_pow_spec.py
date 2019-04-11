import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#global parameters
Mpl = 1.
G = 1./(8*np.pi*Mpl**2)
m1 = 5.5e-6*Mpl
width_chi = 4e-5*Mpl
phipk = 12*Mpl #the location of the peak of the additional gaussian potential
mu = 0.4*Mpl**2 # sigma^2 of the additional gaussian potential
alpha = 0.01

def V(phi,chi):
    return 1./2*m1**2*phi**2*(1 + 0.5*(1 + np.tanh(chi/width_chi))*alpha*np.exp(-(phi - phipk)**2/(2*mu)))

def V_phi(phi,chi):
    eps = 10**-6*Mpl
    return (V(phi+eps/2,chi) - V(phi-eps/2,chi))/eps

def V_chi(phi,chi):
    eps = 10**-6*Mpl
    return (V(phi,chi+eps/2) - V(phi,chi-eps/2))/eps

def d_dlna(var,lna,paras):
    H,phi,psi,chi,xi = var  #psi = d phi / d  lna;  xi = d chi / d  ln a
    m1,width_chi = paras
    return [(8*np.pi*G/(3*H))*(V(phi,chi) - H**2*(psi**2 + xi**2)) - H,
            psi,
            -2*psi - V_phi(phi,chi)/H**2 - (8*np.pi*G*psi/3)*(V(phi,chi)/H**2 - psi**2 - xi**2),
            xi,
            -2*xi - V_chi(phi,chi)/H**2 - (8*np.pi*G*xi/3)*(V(phi,chi)/H**2 - psi**2 - xi**2)]

def thirdH(phi,psi,chi,xi):
    return np.sqrt(2*V(phi,chi)/(3/(4*np.pi*G) - (psi**2 + xi**2)))

phi0 = 18.*Mpl
psi0 = 0
chi0 = 2*width_chi
xi0 = 1*width_chi
H0 = thirdH(phi0,psi0,chi0,xi0)
paras = [m1,width_chi]
var0 = [H0,phi0,psi0,chi0,xi0]

lnaStop = 50
num_of_pts = 50000
lnavals = np.linspace(0,lnaStop,num_of_pts)

sol = odeint(d_dlna,var0,lnavals,args=(paras,))
Hvals = sol[:,0]
phivals = sol[:,1]
psivals = sol[:,2]
chivals = sol[:,3]
xivals = sol[:,4]
Hcheck = np.empty(num_of_pts)

plt.plot(lnavals,phivals)
plt.show()
plt.plot(lnavals,chivals)
plt.show()

for i in range(num_of_pts):
    Hcheck[i] = thirdH(phivals[i], psivals[i], chivals[i], xivals[i])/Hvals[i]-1.
    
# check the accuracy of background solution
plt.plot(lnavals,Hcheck)
plt.show()

# perturbation fields
epsvals = 0.5*(psivals**2 + xivals**2)/Mpl**2

def V_phi_phi(phi,chi):
    eps = 10**-6*Mpl
    return (V_phi(phi+eps/2,chi) - V_phi(phi-eps/2,chi))/eps

def V_phi_chi(phi,chi):
    eps = 10**-6*Mpl
    return (V_phi(phi,chi+eps/2) - V_phi(phi,chi-eps/2))/eps

def V_chi_chi(phi,chi):
    eps = 10**-6*Mpl
    return (V_chi(phi,chi+eps/2) - V_chi(phi,chi-eps/2))/eps


c_11_vals = V_phi_phi(phivals,chivals)/Hvals**2 + (1./Hvals**2/Mpl**2)*(2.*psivals*V_phi(phivals,chivals)) + (3 - epsvals)*psivals**2/Mpl**2

c_12_vals = V_phi_chi(phivals,chivals)/Hvals**2  + (1./Hvals**2/Mpl**2)*(psivals*V_chi(phivals,chivals) + xivals*V_phi(phivals,chivals))+ (3 - epsvals)*psivals*xivals/Mpl**2

c_21_vals = c_12_vals

c_22_vals = V_chi_chi(phivals,chivals)/Hvals**2 + (1./Hvals**2/Mpl**2)*(2*xivals*V_chi(phivals,chivals)) + (3 - epsvals)*xivals**2/Mpl**2

# pert_start = num_of_pts//20    # start calculate perturbation at lnavals[pert_start]
pert_start = 0
while chivals[pert_start] >= 0 and pert_start < num_of_pts - 1:
    pert_start += 1
pert_start -= num_of_pts//lnaStop
print(pert_start)
avals = np.exp(lnavals)
dotavals = np.exp(lnavals)*Hvals
mk = 100   # k = mk*a*H
k = mk*Hvals[pert_start]*avals[pert_start]

# get_val_for_given_lna, return arr(lna)


def get_val(arr,x):
    return np.interp(x,lnavals,arr)


def d_dlna_ij(var,lna,paras):
    RP11, RQ11, IP11, IQ11, RP22, RQ22, IP22, IQ22 = var      #P = delta phi   Q = d(delta phi)/dlna
    m1, width_chi = paras
    eps = get_val(epsvals,lna)
    a = np.exp(lna)
    H = get_val(Hvals,lna)
    c_11 = get_val(c_11_vals,lna)
    c_22 = get_val(c_22_vals,lna)
    c_12 = get_val(c_12_vals,lna)
    c_21 = get_val(c_21_vals,lna)
    result = [RQ11,
              (eps - 3)*RQ11 - (k**2/(a**2*H**2) + c_11)*RP11,
              IQ11,
              (eps - 3)*IQ11 - (k**2/(a**2*H**2) + c_11)*IP11,
              RQ22,
              (eps - 3)*RQ22 - (k**2/(a**2*H**2) + c_22)*RP22,
              IQ22,
              (eps - 3)*IQ22 - (k**2/(a**2*H**2) + c_22)*IP22]
    return result


RP11_0 = RP22_0 = k/(np.sqrt(2)*avals[pert_start])
IP11_0 = IP22_0 = 0
RQ11_0 = RQ22_0 = 0
IQ11_0 = IQ22_0 = -mk*k/(np.sqrt(2)*avals[pert_start])

ptb_init = [RP11_0,RQ11_0,IP11_0,IQ11_0,RP22_0,RQ22_0,IP22_0,IQ22_0]

ptb_sol = odeint(d_dlna_ij,ptb_init,lnavals[pert_start:],args=(paras,))

Re_delta_phi = ptb_sol[:,0]
Im_delta_phi = ptb_sol[:,2]
delta_phi = np.sqrt(Re_delta_phi**2 + Im_delta_phi**2)

Re_delta_chi = ptb_sol[:,4]
Im_delta_chi = ptb_sol[:,6]
delta_chi = np.sqrt(Re_delta_chi**2 + Im_delta_chi**2)

plt.plot(lnavals[pert_start:],delta_chi)
plt.show()


# power spectrum
P_delta_phi_11 = 1./(2*np.pi**2)*delta_phi**2
P_delta_phi_22 = 1./(2*np.pi**2)*delta_chi**2

# vec{omega}
omega_1 = psivals[pert_start:]/np.sqrt(psivals[pert_start:]**2 + xivals[pert_start:]**2)
omega_2 = xivals[pert_start:]/np.sqrt(psivals[pert_start:]**2 + xivals[pert_start:]**2)

# ACP spectrum
P_R = 1./(2*epsvals[pert_start:])*(omega_1**2*P_delta_phi_11 + omega_2**2*P_delta_phi_22)

# vec{s}
s_1 = omega_2
s_2 = -omega_1

# isocurvature power spectrum
P_S = 1./(2*epsvals[pert_start:])*(s_1**2*P_delta_phi_11 + s_2**2*P_delta_phi_22)

print(P_delta_phi_22[(num_of_pts-pert_start)*3//5],P_delta_phi_22[(num_of_pts-pert_start)*4//5])
print(P_delta_phi_11[(num_of_pts-pert_start)*3//5],P_delta_phi_11[(num_of_pts-pert_start)*4//5])
print(P_R[(num_of_pts-pert_start)*3//5],P_R[(num_of_pts-pert_start)*4//5])
print(P_S[(num_of_pts-pert_start)*3//5],P_S[(num_of_pts-pert_start)*4//5],P_S[-1])

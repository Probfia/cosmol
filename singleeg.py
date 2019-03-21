import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Definitions
G=1 #1/Mpl^2
m = 10**-6 #Mpl
def H(phi,psi):
    return np.sqrt(8*np.pi*G/3*(1./2*m**2*phi**2))/np.sqrt(1-1./2*psi**2)
def dHdphi(phi,psi):
    eps = 10**-5
    return (H(phi+eps,psi)-H(phi,psi))/eps
def dHdpsi(phi,psi):
    eps = 10**-5
    return (H(phi,psi+eps)-H(phi,psi))/eps
#Calculate derivatives
def d_dlnus(var, lna, params):
    phi, psi = var # unpack current values of the variables
    H0, m = params # unpack parameters
    derivs = [psi, -1./(1+1/H(phi,psi)*dHdpsi(phi,psi)*psi) *((1/H(phi,psi)*dHdphi(phi,psi)*psi + 3)*psi+m**2/H(phi,psi)**2*phi)]
    return derivs

#Setup
phi0 = 17
psi0 = 0
H0 = H(phi0,psi0)
params = [H0, m]
var0 = [phi0, psi0]
#Calculate over this u=lna
lnaStop = 2000
num_points = 10000
lna = np.linspace(0, lnaStop, num_points)
#Call solver
sol = odeint(d_dlnus, var0, lna, args=(params,))
#Solutions
phi = sol[:,0]

plt.plot(lna,phi)
plt.show()

with open('sfsolution2.txt','w') as fobj:
    for i in range(len(lna)):
        fobj.write(str(lna[i]) + ' ' + str(phi[i]) +'\n')
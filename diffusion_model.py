import numpy as np
from matplotlib import pyplot as plt
from scipy import special


x = 6
dx = 0.01
dx2 = dx**2
nx = int(x/dx)
cAg = np.zeros(nx)
cAg0 = 0.8
t = 500000
dt = 0.1
nt = int(t/dt)
dtdx2 = dt/dx2
DAg = 3*(10**-9)*dtdx2
xgrid = np.linspace(0, x, nx)
# for i in range(nx):
#         cAg[i] = cAg0*(special.erf(nx-(9*i))+1)/2
      
def diff(x, Dx):
    xd = np.zeros(nx)
    xd[0] = Dx*(x[1]-x[0])
    xd[-1] = Dx*(x[-2]-x[-1])
    xL = np.concatenate((x[1:], x[:1]))
    xR = np.concatenate((x[-1:], x[:-1]))
    xd[1:-1] = Dx*(xL[1:-1]+xR[1:-1]-2*x[1:-1]) 
    return xd
cAg[270:330] = cAg0
cAgt0 = cAg

for i in range(nt):
    # cAg[270:330] = cAg0
    if i == 20000:
        cAgt1 = cAg
    elif i == 100000:
        cAgt2 = cAg
    elif i == 4900000:
        cAgt3 = cAg
    elif i == 4000000:
        cAgt4 = cAg
    cAg = cAg + diff(cAg, DAg) 

plt.plot(xgrid, cAgt0,"-b", label='t = 0', linewidth=3)
plt.plot(xgrid, cAgt1, "-",label='t = 2000', linewidth=3, color='orange')
plt.plot(xgrid, cAgt2, "-g",label='t = 10000', linewidth=3)
plt.plot(xgrid, cAgt3, "-r", label='t = 30000', linewidth=3)
# plt.plot(xgrid, cAgt4, "-",label='t = 40000', linewidth=3, color='purple')
# plt.legend()
plt.show()


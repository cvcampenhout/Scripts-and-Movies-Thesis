import numpy as np
import time
from scipy import special
from numba import njit
import numba as nb
from joblib import Parallel, delayed
import multiprocessing
#Measuring run time
start_time = time.time()

#Defining the box
x = 0.005
y = 0.05
dx = 0.00005
dy = 0.00005
dx2 = dx**2
dy2 = dy**2
nx = int(x/dx)
ny = int(y/dy)
xgrid = np.linspace(0, x, nx)
ygrid = np.linspace(0, y, ny)
dt = .01
dtdx2 = dt/dx2
#reaction/diffusion constants
DAg = 3*(10**-9)*dt
Dsol = 3*(10**-9)*dt
Ddich = 0*(10**-9)*dt
ksalt = 0.001*dt
knuc = (0.001/1000)*dt 
kgrowth = 0.01*10000*dt
cthresh0 = 0.005
randomgrid = np.random.uniform(0.9,1.1, size=((ny,nx)))
cthresh = np.ones((ny,nx))*cthresh0*randomgrid
#starting concentrations
cAg0 = 0.80
cdich0 = 0.2

volinner = 1
volouter = 10

@njit
def roller_left(a):
    b = np.zeros((ny,nx))
    for i in nb.prange(ny):
        b[i, 1:] = a[i, :(nx - 1)]
        b[i, :1] = a[i, (nx - 1):]
    return b
@njit
def roller_right(a):
    b = np.zeros((ny,nx))
    for i in nb.prange(ny):
        b[i, :(nx - 1)] = a[i, 1:]
        b[i, (nx - 1):] = a[i, :1]
    return b
@njit
def roller_bot(a):
    b = np.zeros((ny,nx))
    for i in nb.prange(nx):
        b[:(ny - 1), i] = a[1:, i]
        b[(ny-1):, i] = a[:1, i]
    return b   
@njit
def roller_top(a):
    b = np.zeros((ny,nx))
    for i in nb.prange(nx):
        b[1:, i] = a[:(ny - 1), i]
        b[:1, i] = a[(ny-1):, i]
    return b   

@njit
def diff(x, Dx):
    xd = np.zeros((ny,nx))
    xT = roller_top(x)
    xB = roller_bot(x)
    xL = roller_left(x)
    xR = roller_right(x)
    xd[1:-1] = Dx*((xL[1:-1]+xR[1:-1]-2*x[1:-1])/dx2 + (xT[1:-1]+xB[1:-1]-2*x[1:-1])/dy2)
    xd[0] = Dx*((x[1]-x[0])/dy2)
    xd[-1] = Dx*((x[-2]-x[-1]))/dy2
    return xd

@njit
def growth_check(x):
    xd = np.zeros((ny,nx))
    xT = roller_top(x)
    xB = roller_bot(x)
    xL = roller_left(x)
    xR = roller_right(x)

    return (xT+xB+xL+xR)>0.001



@njit
def simulator(cdich, cAg, csol, cNP, act_time, cthresh, rate_mid, tgrid, Dsol): 
    for i in tgrid:   
        act_time = i
        cAg = cAg + diff(cAg, DAg) - cAg*cdich*ksalt 
        cdich = cdich +diff(cdich, Ddich) - cAg*cdich*ksalt
        csol = csol + diff(csol, Dsol) + cAg*cdich*ksalt - knuc*csol*(csol>cthresh) - csol*cNP*kgrowth  - csol*growth_check(cNP)*kgrowth
        cNP = cNP + knuc*csol*(csol>cthresh) + csol*cNP*kgrowth  + csol*growth_check(cNP)*knuc
        rate = rate_mid
        channel_open = rate*act_time
        channel_front = int(channel_open/y) + int(ny/10)
        cAg[:channel_front] = cAg0* (volouter/(((channel_front/ny)*volinner)+volouter))

    return cNP, cAg

def init_run(Dsol0):
    act_time = 0
    Dsol = Dsol0*(10**-9)*dt
    t = 5000 #t in seconds
    nt = int(t/dt)
    tgrid = np.linspace(0,t,nt)
    cAg = np.zeros((ny,nx))
    cdich = np.zeros((ny,nx))
    csol = np.zeros((ny,nx))
    cNP = np.zeros((ny,nx))
    rate_mid = 0.004
    randomgrid = np.random.uniform(0.9,1.1, size=((ny,nx)))
    cthresh = np.ones((ny,nx))*cthresh0*randomgrid
    #Filling cA and canion np arrays with starting concentrations (erf to provide less of a sharp interface)
    for i in range(ny):
        cAg[i] = cAg0*(special.erf(ny-(9*i))+1)/2
        cdich[i] = cdich0*(special.erf((9*i)-ny)+1)/2

    name = f"19102023_long2Dpump_cdich={Dsol0}"

    pattern_final, silver_conc = simulator(cdich, cAg, csol, cNP, act_time, cthresh, rate_mid, tgrid, Dsol)
    np.savetxt(f'{name}_pattern.csv', pattern_final, delimiter=',')
    np.savetxt(f'{name}_silver.csv', silver_conc, delimiter=',')
    # np.savetxt(f'{name}_test.csv', cAg, delimiter=',')
    #Printing the runtime
    print(f"Run took: {time.time() - start_time:.2f} seconds") 


Dsol_list = [9, 15]
# nuc_list = [0.01]
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(init_run)(i) for i in Dsol_list)

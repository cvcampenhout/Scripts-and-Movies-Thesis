from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from scipy import special
import csv

def diff(x, Dx):
    xd = np.zeros(nx)
    xd[0] = Dx*(x[1]-x[0])
    xd[-1] = Dx*(x[-2]-x[-1])
    xL = np.concatenate((x[1:], x[:1]))
    xR = np.concatenate((x[-1:], x[:-1]))
    xd[1:-1] = Dx*(xL[1:-1]+xR[1:-1]-2*x[1:-1]) 
    return xd

#Funcanimation requires an initializing function that provides a starting frame, returns the arrays that will hold data to plot
def init():
    line2.set_ydata(np.ma.array(xgrid*100, mask=True))  
    line3.set_ydata(np.ma.array(xgrid*100, mask=True)) 
    return line2, line3

def animtime(n):
    global cCa
    global calg
    global xlist
    global cdich
    global cAg
    global csol
    global cNP
    global cCaalg


    for i in range(nframes):

        #reaction of A with alginate, induces "shrink"
        cCa = cCa + diff(cCa, DCa) - cCa*calg*calg*kCaalg + knegCaalg*cCaalg
        calg = calg - 2*cCa*calg*calg*kCaalg + 2*knegCaalg*cCaalg
        cCaalg = cCaalg + cCa*calg*calg*kCaalg - knegCaalg*cCaalg
        
        #reaction between silver and dichromate
        cAg = cAg + diff(cAg, DAg) - 2*cAg*cAg*cdich*ksalt 
        cdich = cdich - cAg*cAg*cdich*ksalt
        csol = csol + diff(csol, Dsol) + cAg*cAg*cdich*ksalt - knuc*csol*(csol>cthresh) - csol*cNP*kgrowth
        cNP = cNP + knuc*csol*(csol>cthresh) + csol*cNP*kgrowth

        # #channel opens when Ca + alg -> Ca-Alg exceeds a threshold; cCaalg>(0.010)
        # #np.where returns the indeces of the requested <; we only look at the last value to give the location of the front

        #channel opens at calg<threshold
        channel_open = np.where(cCaalg>(0.010))
        # ballistic with opening channel
        channel_front = channel_open[0][-1]

        # # diffusive with rapid mixing in reservoir:
        # channel_front = 100

        #Setting outer concentration over entire sample, accounting for dillution
        cCa[:channel_front] = cCa0* (volouter/(((channel_front/nx)*volinner)+volouter))
        cAg[:channel_front] = cAg0* (volouter/(((channel_front/nx)*volinner)+volouter))
        
    cCaalg_eq = cCaalg*10
    line2.set_ydata(cdich)
    line3.set_ydata(cAg)
    line4.set_ydata(csol)
    line5.set_ydata(cNP)
    line6.set_ydata(cCaalg_eq)

    # xfrontloc = np.where(cAg>0.02)
    # xfrontindex = xfrontloc[0][-1]
    # #following the precipitation front as well as diffusive zone lengths
    # if np.sum(cNP) > 0.1:
    #     preciploc = np.where(cNP>0.02)
    #     precip_index = preciploc[0][-1]
    #     g_index = (precip_index - channel_front)
    #     g = (g_index+1)/nx * x * 1000
    #     glist.append(g)
    # xfront = (xfrontindex+1)/nx * x * 1000
    # xlist.append(xfront)

    return line2, line3, line4, line5

#Measuring run time
start_time = time.time()
#Defining the box
x = 0.1
dx = 0.00001
dx2 = dx**2
nx = int(x/dx)
xgrid = np.linspace(0, x, nx)
t = 500000 #t in seconds
dt = .01
nt = int(t/dt)
dtdx2 = dt/dx2

#Starting concentrations
cCa0 = 6
cAg0 = 0.80
cCaalg0 = 0.05
#Estimated conc of NaAlg = 0.05 M (for a 1w% solution with avg MW = 222, http://www.fao.org/3/W6355E/w6355e0x.htm)
calg0 = 0.05
cdich0 = 0.26

cthresh0 = 0.1
cthresh = np.ones(nx)*cthresh0

#Binding affinities (to alginate, from: Biol.  Pharm.  Bull.39,  1893–1896  (2016)) 
bCa = 779.2

#reaction/diffusion constants
DCa = 3*(10**-9)*dtdx2
DAg = 3*(10**-9)*dtdx2
Dsol = 0.1*(10**-9)*dtdx2
kshrink = 0.000075*dt
ksalt = 0.001*dt
knuc = (0.001/1000)*dt 
kgrowth = 0.01*10000*dt
kCaalg = kshrink
knegCaalg = kCaalg/bCa
volouter = 100
volinner = 1
for ix in [1, 3, 6]:
    cCa0 = ix
    #Providing empty np arrays
    cCa = np.zeros(nx)
    calg = np.zeros(nx)
    cAg = np.zeros(nx)
    cdich = np.zeros(nx)
    csol = np.zeros(nx)
    cNP = np.zeros(nx)
    cCaalg = np.zeros(nx)
    cCaalg_eq = np.zeros(nx)
    #Filling cA and canion np arrays with starting concentrations (erf to provide less of a sharp interface)
    for i in range(nx):
        cCa[i] = cCa0*(special.erf(nx-(9*i))+1)/2
        cAg[i] = cAg0*(special.erf(nx-(9*i))+1)/2
        calg[i] = calg0*(special.erf((9*i)-nx)+1)/2
        cdich[i] = cdich0*(special.erf((9*i)-nx)+1)/2
        cCaalg[i] = cCaalg0*(special.erf(nx-(9.9*i))+1)/2

    #ANIMATION SETUP
    #anitgrid for animation (every nth frame)
    nframes = 10000
    anitgrid = np.linspace(0,t, int(nt/nframes))
    tgrid = np.linspace(0,t,int(nt/nframes))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.tick_params(labelsize=20)
    ax.set_xlabel("x (m)", fontsize=20)
    ax.set_ylabel("Concentration (M)", fontsize=20)

    #providing a y limit, can use to zoom in animation
    plt.ylim(0,0.9)
    plt.xlim(0.007, 0.09)
    #providing starting data before starting the animation
    # line1, = ax.plot((xgrid), cCa, color="#DC143C",label=r"$Ca^{2+}$")
    line2, = ax.plot((xgrid), cdich, color="#afabab", label="reductant",linewidth=3)
    line3, = ax.plot((xgrid), cAg, color="#767171", label=r"$AgNO_{3}$",linewidth=3)
    line4, = ax.plot((xgrid), csol, color="#000000", label="Ag sol",linewidth=3)
    line5, = ax.plot((xgrid), cNP, color="#df4b07", label="Ag NP's",linewidth=2)
    line6, = ax.plot((xgrid), cCaalg_eq, color="#00b0f0", label="Ca-alg",linewidth=3)

    xlist = []
    glist = []
    randomgrid = np.random.uniform(0.99,1.01, size=(nx))
    cthresh = np.ones(nx)*cthresh0*randomgrid

    #Actually running funcanimation and saving the results as anim
    anim = FuncAnimation(fig, animtime, frames=anitgrid, init_func=init, interval=0, blit=True)
    name = f"14022024_MRG-PNAS_modellongruncCa={cCa0}"
    #Saving the animation as an mp4 file, fps=20, higher can be used to speed up video
    #dpi sets image quality, vcodes and libx264 define what codec etc. to use (copied from somewhere and this works)
    anim.save(f'{name}.mp4', fps=25, extra_args=['-vcodec', 'libx264'], dpi=300)
    plt.close()
    #Printing the runtime
    print(f"Run took: {time.time() - start_time:.2f} seconds") 



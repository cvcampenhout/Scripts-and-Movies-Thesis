from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import special

#Measuring run time
start_time = time.time()

#Defining the box
x = 0.1
dx = 0.0001
dx2 = dx**2
nx = int(x/dx)
xgrid = np.linspace(0, x, nx)
t = 1040000
dt = .01
nt = int(t/dt)
dtdx2 = dt/dx2

#Starting concentrations

cAg0 = 0.80
#Estimated conc of NaAlg = 0.05 M (for a 1w% solution with avg MW = 222, http://www.fao.org/3/W6355E/w6355e0x.htm)
calg0 = 0.05
cdich0 = 0.28

cthresh0 = 0.24
randomgrid = np.random.uniform(0.9,1.1, size=(nx))
cthresh = np.ones(nx)*cthresh0*randomgrid


#reaction/diffusion constants

DAg = 3*(10**-9)*dtdx2
DNP = 0.1*(10**-9)*dtdx2
Dsol = 0.1*(10**-9)*dtdx2
kshrink = 0.001*dt
kdich = kshrink*100
knuc = kshrink/1000
kgrowth = kshrink*10000


volinner = 1
volouter = 10

#Providing empty np arrays

cAg = np.zeros(nx)
cdich = np.zeros(nx)
csol = np.zeros(nx)
cNP = np.zeros(nx)


for i in range(nx):
    cAg[i] = cAg0*(special.erf(nx-(9*i))+1)/2
    cdich[i] = cdich0*(special.erf((9*i)-nx)+1)/2


def diff(x, Dx):
    xd = np.zeros(nx)
    xd[0] = Dx*(x[1]-x[0])
    xd[-1] = Dx*(x[-2]-x[-1])
    xL = np.concatenate((x[1:], x[:1]))
    xR = np.concatenate((x[-1:], x[:-1]))
    xd[1:-1] = Dx*(xL[1:-1]+xR[1:-1]-2*x[1:-1]) 
    return xd

def deltareact(x, y, kxy, ordx, ordy):
    return kxy*(x**ordx)*(y**ordy)


#ANIMATION SETUP

#anitgrid for animation (every nth frame)
nframes = 10000
anitgrid = np.linspace(0,t, int(nt/nframes))
tgrid = np.linspace(0,(t/60),int(nt/nframes))

#Funcanimation requires an initializing function that provides a starting frame, returns the arrays that will hold data to plot
def init():
    # line1.set_ydata(np.ma.array(xgrid, mask=True))
    line2.set_ydata(np.ma.array(xgrid, mask=True))
    return line2,

#Funcanimation animates a function, and gives it "frames=" as input
#now gives array [0,nframes,nframes*2,...]  as input, and does nframes computations for each iterations, effectively animating only every nth frame
#which reduces animation time n-fold
#total number of tsteps does not change because of anitgrid scaling inverse with nframes, opposite of the for loop

def animtime(n):
    
    global cdich
    global cAg
    global csol
    global cNP

    for i in range(nframes):
        cAg[0] = cAg0
        cAg = cAg + diff(cAg, DAg) - 2*cAg*cAg*cdich*kdich 
        cdich = cdich - cAg*cAg*cdich*kdich
        csol = csol + diff(csol, Dsol) + cAg*cAg*cdich*kdich - knuc*csol*(csol>cthresh) - csol*cNP*kgrowth
        cNP = cNP + knuc*csol*(csol>cthresh0) + csol*cNP*kgrowth
    line2.set_ydata(cdich)
    line3.set_ydata(cAg)
    line4.set_ydata(csol)
    line5.set_ydata(cNP)


    return line2,line3,line4

#defining the size of the eventual animation
fig, ax = plt.subplots(figsize=(12, 7))
ax.tick_params(labelsize=20)
ax.set_xlabel("x (m)", fontsize=20)
ax.set_ylabel("Concentration (M)", fontsize=20)

#providing a y limit, can use to zoom in animation
plt.ylim(0,cdich0*3)
plt.xlim(0.007, 0.09)
#providing starting data before starting the animation
# line1, = ax.plot((xgrid), cCa, color="#DC143C",label=r"$Ca^{2+}$")
line2, = ax.plot((xgrid), cdich, color="#afabab", label="dichromate",linewidth=3)
line3, = ax.plot((xgrid), cAg, color="#767171", label=r"$AgNO_{3}$",linewidth=3)
line4, = ax.plot((xgrid), csol, color="#000000", label="Ag sol",linewidth=3)
line5, = ax.plot((xgrid), cNP, color="#df4b07", label="Ag NP's",linewidth=2)

cdich0 = 0.28
xlist = []
randomgrid = np.random.uniform(0.99,1.01, size=(nx))
cthresh = np.ones(nx)*cthresh0*randomgrid
#Providing empty np arrays
cAg = np.zeros(nx)
cdich = np.zeros(nx)
csol = np.zeros(nx)
cNP = np.zeros(nx)

#Filling cA and canion np arrays with starting concentrations (erf to provide less of a sharp interface)
for i in range(nx):
    cAg[i] = cAg0*(special.erf(nx-(9*i))+1)/2
    cdich[i] = cdich0*(special.erf((9*i)-nx)+1)/2

#Actually running funcanimation and saving the results as anim
anim = FuncAnimation(fig, animtime, frames=anitgrid, init_func=init, interval=0, blit=True)
name = f"14022024_RDmodel_diffusive"
#Saving the animation as an mp4 file, fps=20, higher can be used to speed up video
#dpi sets image quality, vcodes and libx264 define what codec etc. to use (copied from somewhere and this works)
anim.save(f'{name}.mp4', fps=25, extra_args=['-vcodec', 'libx264'], dpi=300)
plt.close()
#Printing the runtime
print(f"Run took: {time.time() - start_time:.2f} seconds") 



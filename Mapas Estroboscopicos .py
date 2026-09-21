import numpy as np
from numpy import sin, cos,pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#----------------------------------------------------
#System parameters
#----------------------------------------------------
m=1
k=1
l=1
F=1
om=2/3*pi
T=2*pi /om

#----------------------------------------------------
#Initial conditions
#----------------------------------------------------
x0=2
v0=0

#----------------------------------------------------
#Methodparameters
#----------------------------------------------------
Trans=0
Nperiods=5000
tmax=Nperiods * T
dt=0.001
#----------------------------------------------------
#Dynamics:ForcedDuffing oscillator
#----------------------------------------------------
def dyn(t,y):
    x,v= y
    dx=v
    dv=F*cos(om*t)/m+(k/m)*x-(l/m)*x**3
    return np.array([dx,dv])
#----------------------------------------------------
#Fourth-orderRunge-Kutta method
#----------------------------------------------------
def rk4(f,t,y,h):
    k1=h * f(t,y)
    k2=h * f(t+h/2,y+k1/2)
    k3=h * f(t+h/2,y+k2/2)
    k4=h * f(t+h,y+k3)
    return y+(k1+2*k2+2*k3+k4)/6
#----------------------------------------------------
#IntegrationusingRK4
#----------------------------------------------------
n=int(tmax/dt)
t=np.linspace(0,n*dt,n+1)
y=np.empty((n+1,2))
y[0]=[x0,v0]
for i in range(n):
    y[i+1]=rk4(dyn,t[i],y[i],dt)
#----------------------------------------------------
#Separatevariablesafter integration
#----------------------------------------------------
x,v=y[:,0],y[:,1]
#----------------------------------------------------
#Figure setting-Stroboscopicpoints
#----------------------------------------------------
fig,ax=plt.subplots(figsize=(8,6))
PE=[]
for j in range(Trans,Nperiods+1):
    tj=j*T
    index=np.argmin(np.abs(t-tj))
    PE.append([x[index],v[index]])
PE=np.array(PE)

#----------------------------------------------------
#Stroboscopicmap
#----------------------------------------------------
ax.scatter(PE[:,0],PE[:, 1],s=3,color='green')
ax.set_xlabel(r'$x$',fontsize=22)
ax.set_ylabel(r'$\dot{x}$',fontsize=22)
#ax.tick_params(axis='both',labelsize=18)
#ax.grid(alpha=0.3)
ax.set_title('Stroboscopic map',fontsize=20)
ax.set_box_aspect(0.65)

#----------------------------------------------------
#Figureformat
#----------------------------------------------------
plt.tight_layout()
plt.savefig("Stroboscopic2.pdf",format="pdf",bbox_inches="tight")
plt.show()
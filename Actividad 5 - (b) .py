import numpy as np
from numpy import sin, cos, pi, abs
import matplotlib.pyplot as plt
from fractions import Fraction

# ----------------------------------------------------
# System parameters
# ----------------------------------------------------
d = 0.1
a = 2.0
b = 2.0 
om = 1.2
T = 2*pi/om


# ----------------------------------------------------
# Initial conditions
# ----------------------------------------------------

x0 = 1
v0 = 1

# ----------------------------------------------------
# Bifurcation parameter: gamma
# ----------------------------------------------------

gamma_min = 0.1
gamma_max = 7.0
dgamma = 0.001
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)

# ----------------------------------------------------
# Initial state
# ----------------------------------------------------

x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
y = np.concatenate([x, v])

# ----------------------------------------------------
# Numerical method parameters
# ----------------------------------------------------

Trans = 150
Nkeep = 50
steps_per_T = 150
dt = T / steps_per_T

# ----------------------------------------------------
# Dynamics
# ----------------------------------------------------

def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -d*v + a*x - b*x**3 + gamma_values*cos(om*t)
    return np.concatenate([dx, dv])

# ----------------------------------------------------
# Fourth-order Runge-Kutta method
# ----------------------------------------------------

def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# ----------------------------------------------------
# Stroboscopic storage
# ----------------------------------------------------

x_strobe = np.empty((Nkeep, n_orbits))
v_strobe = np.empty((Nkeep, n_orbits))
save_index = 0

# ----------------------------------------------------
# Integration
# ----------------------------------------------------

total_periods = Trans + Nkeep
total_steps = total_periods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# ----------------------------------------------------
# Bifurcation diagram & Figure format
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(np.full(Nkeep, gamma_values[i]),
              v_strobe[:,i], s=0.5, color='blue', linewidths=0, rasterized=True)
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\dot{x}$',fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$(b)$',
        transform=ax.transAxes, ha='left', va='top', 
        fontsize=14, math_fontfamily='stix')
ax.set_xlim(gamma_min, gamma_max)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('Bifurcacion x punto.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
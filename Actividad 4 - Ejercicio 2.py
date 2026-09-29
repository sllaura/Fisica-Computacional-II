import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ----------------------------------------------------
# Parámetros del sistema (oscilador de Duffing forzado y amortiguado)
# Con la convención "-alpha x", alpha = -1 da rigidez lineal positiva
# (un solo pozo de potencial).
# ----------------------------------------------------
delta = 0.02
alpha = -1.0
beta = 5.0
gamma = 8.0
om = 0.5
T = 2 * pi / om

# Condiciones iniciales aleatorias (muchas más órbitas)
rng = np.random.default_rng(0)
n_orbits = 1500
x0_flat = rng.uniform(-2, 2, n_orbits)
v0_flat = rng.uniform(-2, 2, n_orbits)
y = np.concatenate([x0_flat, v0_flat])

Trans = 200
Nperiods = 1700
steps_per_T = 200
dt = T / steps_per_T

def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dv = gamma * cos(om * t) + alpha * x - beta * x**3 - delta * v
    return np.concatenate([v, dv])

def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

n_saved = Nperiods - Trans
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))

save_index = 0
for step in range(Nperiods * steps_per_T):
    y = rk4(dyn, step * dt, y, dt)
    if (step + 1) % steps_per_T == 0:
        period = (step + 1) // steps_per_T
        if period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# Colores por órbita (como en la figura de referencia)
cmap = ListedColormap(['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
color_idx = np.tile(np.arange(n_orbits) % 6, (n_saved, 1))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_strobe.ravel(), v_strobe.ravel(), c=color_idx.ravel(), cmap=cmap,
           s=1, alpha=0.6, linewidths=0, rasterized=True)
ax.set_xlim(1.05, 1.76)
ax.set_ylim(-2.8, 2.8)
ax.set_xlabel(r'$x$', fontsize=22)
ax.set_ylabel(r'$\dot{x}$', fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('mapa_estroboscopico_ejercicio2.pdf', format='pdf',
            bbox_inches='tight', dpi=300)
plt.show()
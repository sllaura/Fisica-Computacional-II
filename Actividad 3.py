import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Parámetros del sistema (péndulo simple forzado sin amortiguamiento)
#   theta'' + (g/l) sin(theta) = (F0 / (m l)) cos(w t)
# ----------------------------------------------------
ml = 1.0        # m*l
gl = 1.0        # g/l
F0 = 0.01
om = 2 / pi
T = 2 * pi / om

# ----------------------------------------------------
# Condiciones iniciales: malla en (theta0, v0)
# theta0 cubre todo el rango -pi..pi y v0 un rango amplio de
# velocidades angulares para reproducir islas de resonancia,
# órbitas cerradas centrales y el mar caótico de los bordes.
# ----------------------------------------------------
x0_values = np.arange(-pi, pi + 0.01, 0.35)
v0_values = np.arange(-2.3, 2.3 + 0.01, 0.4)
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()
n_orbits = len(x0_flat)
y = np.concatenate([x0_flat, v0_flat])

# ----------------------------------------------------
# Parámetros del método
# ----------------------------------------------------
Trans = 0
Nperiods = 400
steps_per_T = 150
dt = T / steps_per_T

# ----------------------------------------------------
# Dinámica: péndulo simple forzado
# theta' = v
# v'     = (F0/ml) cos(w t) - (g/l) sin(theta)
# ----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = (F0 / ml) * cos(om * t) - gl * sin(x)
    return np.concatenate([dx, dv])

# ----------------------------------------------------
# Integrador de Runge-Kutta de orden 4
# ----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# ----------------------------------------------------
# Almacenamiento estroboscópico
# ----------------------------------------------------
n_saved = Nperiods - Trans + 1
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))

save_index = 0
if Trans == 0:
    x_strobe[save_index] = y[:n_orbits]
    v_strobe[save_index] = y[n_orbits:]
    save_index += 1

# ----------------------------------------------------
# Integración
# ----------------------------------------------------
total_steps = Nperiods * steps_per_T
for step in range(total_steps):
    current_time = step * dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period >= max(1, Trans):
            # Envolver theta en (-pi, pi] para que el mapa quede acotado,
            # tal como en la figura de referencia
            theta = y[:n_orbits]
            theta = (theta + pi) % (2 * pi) - pi
            y[:n_orbits] = theta
            x_strobe[save_index] = theta
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# ----------------------------------------------------
# Colores según condición inicial y gráfico del mapa estroboscópico
# ----------------------------------------------------
orbit_colors = plt.cm.turbo(np.linspace(0, 1, n_orbits))

fig, ax = plt.subplots(figsize=(7, 5))
for i in range(n_orbits):
    ax.scatter(x_strobe[:, i], v_strobe[:, i], s=1.2, color=orbit_colors[i],
               linewidths=0, rasterized=True)
ax.set_xlabel(r'$\theta$', fontsize=22)
ax.set_ylabel(r"$\dot{\theta}$", fontsize=22)
ax.set_xlim(-pi, pi)
ax.set_ylim(-2.2, 2.2)
ax.tick_params(axis="both", labelsize=18)
plt.tight_layout()
plt.savefig('mapa_estroboscopico_pendulo.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
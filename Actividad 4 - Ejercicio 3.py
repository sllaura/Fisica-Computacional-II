import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Ejercicio 3 - Mapa estroboscópico (comparar con la Fig. 1(b))
#   x'' + delta x' - alpha x + beta x^3 = gamma cos(w t)
# ----------------------------------------------------

# ----------------------------------------------------
# Parámetros del sistema (oscilador de Duffing forzado y amortiguado)
# OJO: con la convención "-alpha x", la Fig. 1(b) (doble pozo, Holmes 1979)
# se obtiene con alpha = +1. Con alpha = -1 (rigidez lineal positiva) el mapa
# colapsa a un único punto fijo y no aparece el atractor extraño.
# ----------------------------------------------------
delta = 0.15
alpha = 1.0
beta = 1.0
gamma = 0.3
om = 1.0

# Malla de condiciones iniciales
x0_values = np.arange(-1.5, 1.5 + 0.01, 0.5)
v0_values = np.arange(-1.0, 1.0 + 0.01, 0.5)
T = 2 * pi / om

# ----------------------------------------------------
# Condiciones iniciales: malla en (x0, v0)
# ----------------------------------------------------
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()
n_orbits = len(x0_flat)
y = np.concatenate([x0_flat, v0_flat])

# ----------------------------------------------------
# Parámetros del método
#   Trans: periodos que se descartan (régimen transitorio)
#   Nperiods: periodos totales integrados
# ----------------------------------------------------
Trans = 200
Nperiods = 2200
steps_per_T = 300
dt = T / steps_per_T

# ----------------------------------------------------
# Dinámica: oscilador de Duffing forzado y amortiguado
# x' = v
# v' = gamma cos(w t) + alpha x - beta x^3 - delta v
# ----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = gamma * cos(om * t) + alpha * x - beta * x**3 - delta * v
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
# Integración: se guarda (x, v) una vez por periodo T = 2 pi / w,
# solo después de descartar los primeros Trans periodos
# ----------------------------------------------------
total_steps = Nperiods * steps_per_T
for step in range(total_steps):
    current_time = step * dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period >= max(1, Trans):
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# ----------------------------------------------------
# Colores según condición inicial y gráfico del mapa estroboscópico
# ----------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
orbit_colors = [colors[i % len(colors)] for i in range(n_orbits)]

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(x_strobe[:, i], v_strobe[:, i], s=1, color=orbit_colors[i],
               linewidths=0, rasterized=True)
ax.set_xlabel(r'$x$', fontsize=22)
ax.set_ylabel(r"$\dot{x}$", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('mapa_estroboscopico_ejercicio3.pdf', format='pdf',
            bbox_inches='tight', dpi=800)
plt.show()

print("Órbitas:", n_orbits, "| puntos por órbita:", n_saved)
print("x en [%.2f, %.2f], v en [%.2f, %.2f]" %
      (x_strobe.min(), x_strobe.max(), v_strobe.min(), v_strobe.max()))
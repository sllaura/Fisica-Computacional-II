"""
Mapa estroboscopico del pendulo simple forzado sin amortiguamiento.
Version vectorizada con mapa de colores para cada orbita.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from itertools import cycle

# ------------------------- Parametros del sistema -------------------------
ml = 1.0
g_l = 1.0
F0 = 0.01
omega = 2.0 / np.pi

T = 2 * np.pi / omega         # Periodo de la fuerza externa
N_PERIODOS = 150             # Puntos por orbita en el mapa

# ---------------------- Condiciones iniciales (barrido) -------------------
theta0_list = np.linspace(-3.0, 3.0, 20)
theta_dot0_list = np.linspace(-2.2, 2.2, 15)

TH0, THD0 = np.meshgrid(theta0_list, theta_dot0_list)
theta0_flat = TH0.ravel()
theta_dot0_flat = THD0.ravel()
n_orbitas = len(theta0_flat)
print(f"Numero de orbitas: {n_orbitas}")

y0 = np.concatenate([theta0_flat, theta_dot0_flat])

# ---------------------- Ecuacion de movimiento (vectorizada) --------------
def pendulo_forzado_vec(t, y):
    theta = y[:n_orbitas]
    theta_dot = y[n_orbitas:]
    dtheta_dt = theta_dot
    dtheta_dot_dt = -g_l * np.sin(theta) + (F0 / ml) * np.cos(omega * t)
    return np.concatenate([dtheta_dt, dtheta_dot_dt])

# ------------------------------- Integrar ----------------------------------
t_eval = np.arange(0, N_PERIODOS + 1) * T

t0 = time.time()
sol = solve_ivp(
    pendulo_forzado_vec,
    t_span=(0, t_eval[-1]),
    y0=y0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-7,
    atol=1e-9,
)
print(f"Integracion terminada en {time.time()-t0:.1f} s, exito={sol.success}")

theta_sol = sol.y[:n_orbitas, :]        # shape (n_orbitas, N_PERIODOS+1)
theta_dot_sol = sol.y[n_orbitas:, :]

# Identificamos theta = -pi con theta = pi
theta_mod = (theta_sol + np.pi) % (2 * np.pi) - np.pi

# ------------------------------- Graficar ----------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

# Mapa de colores para iterar y pintar cada órbita con un color distinto
colors = plt.cm.turbo(np.linspace(0, 1, n_orbitas))

for i in range(n_orbitas):
    ax.plot(theta_mod[i, :], theta_dot_sol[i, :], '.', 
            color=colors[i], markersize=1.2, alpha=0.9)

# Estilo visual idéntico a la imagen de referencia
ax.set_xlabel(r'$\theta$', fontsize=12)
ax.set_ylabel(r'$\dot{\theta}$', fontsize=12)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-2.2, 2.2)

# Configurar marcas y marcas secundarias (ticks hacia adentro)
ax.tick_params(direction='in', top=True, right=True, labelsize=10)

plt.tight_layout()
plt.savefig("mapa_v3.png", dpi=300)
plt.show()
print("Figura guardada.")

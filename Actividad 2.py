import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
g = 9.81
l = 1.0      # longitud del péndulo
m = 1.0      # masa del péndulo
M_r = 4.0      # masa del bloque
k = 50.0     # constante elástica del resorte

# -----------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------
x0 = 0.2                 # desplazamiento inicial del bloque
v0 = 0.0                 # velocidad inicial del bloque
th0 = np.radians(20.0)   # ángulo inicial del péndulo
ome0 = 0.0                # velocidad angular inicial

# -----------------------------------------------------------
# method parameters
# -----------------------------------------------------------
tmax = 20
dt = 0.01
STRIDE = 2

# -----------------------------------------------------------
# Dynamics: pendulum with sliding support on a spring
# -----------------------------------------------------------
def dyn(t, y):
    x, v, th, ome = y
    s, c = sin(th), cos(th)
    den = M_r/m + s**2

    a_x = ((g*c + l*ome**2)*s - (k/m)*x) / den
    a_th = -(1.0/l) * (g*(1 + M_r/m)*s + c*(l*ome**2*s - (k/m)*x)) / den

    return np.array([v, a_x, ome, a_th])

# -----------------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------------
# Integration using RK4
# -----------------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n*dt, n+1)
y = np.empty((n+1, 4))
y[0] = np.array([x0, v0, th0, ome0])
for i in range(n):
    y[i+1] = rk4(dyn, t[i], y[i], dt)

# -----------------------------------------------------------
# Separate variables after integration
# -----------------------------------------------------------
x = y[:, 0]
v = y[:, 1]
th = y[:, 2]
ome = y[:, 3]

# -----------------------------------------------------------
# Kinematics
# -----------------------------------------------------------
x_block, y_block = x, np.zeros_like(x)
x_bob = x + l*sin(th)
y_bob = -l*cos(th)

# -----------------------------------------------------------
# Figure
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
R = l + max(abs(x).max(), 0.5) + 0.3
ax.set_xlim(-R, R); ax.set_ylim(-R, R/2)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Péndulo con soporte deslizante y resorte (RK4)")
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)

# Pared fija donde se ancla el resorte (rectángulo con rayado)
x_wall = -R + 0.15
wall_width = 0.08
wall = Rectangle((x_wall - wall_width, -R/4), wall_width, R/4 + 0.4,
                  facecolor="white", edgecolor="black", hatch="////", zorder=1)
ax.add_patch(wall)

# Bloque M: rectángulo negro que se desliza horizontalmente
block_w, block_h = 0.35, 0.30
block = Rectangle((x0 - block_w/2, -block_h/2), block_w, block_h,
                    facecolor="navy", edgecolor="black", zorder=4)
ax.add_patch(block)

rod, = ax.plot([], [], "-", lw=1.5, color="black", zorder=6)
bob, = ax.plot([], [], "o", ms=16, color="blue", mec="black", mew=1.2, zorder=7)
trace, = ax.plot([], [], "-", lw=1, alpha=0.5, color="blue", zorder=3)
spring_line, = ax.plot([], [], lw=1.5, color="black", zorder=2)
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=12, color="black")

# -----------------------------------------------------------
# Create Animation
# -----------------------------------------------------------
l0_ref = abs(x_wall)  # longitud de referencia para escalar la amplitud del resorte

def spring_coords(x1, y1, x2, y2, n_coils=12):
    """Genera coordenadas de un resorte helicoidal entre dos puntos (x1,y1) y (x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy) + 1e-9
    s_param = np.linspace(0, 1, 300)
    s = 0.5 * (1 - np.cos(np.pi * s_param))   # distribución no uniforme
    nx = -dy / L
    ny = dx / L
    amp = np.clip(0.08 * np.sqrt(l0_ref / L), 0.04, 0.10) if L > 0 else 0.08
    wave = amp * np.sin(2 * np.pi * n_coils * s)
    xs = x1 + dx * s + nx * wave
    ys = y1 + dy * s + ny * wave
    return xs, ys

def animate(i):
    xb = x_block[i]
    y_pivot = -block_h/2   # la cuerda sale del borde inferior del bloque, no de su centro
    rod.set_data([xb, x_bob[i]], [y_pivot, y_bob[i]])
    bob.set_data([x_bob[i]], [y_bob[i]])
    block.set_xy((xb - block_w/2, -block_h/2))
    trace.set_data(x_bob[:i+1], y_bob[:i+1])
    xs, ys = spring_coords(x_wall, 0.0, xb - block_w/2, 0.0)
    spring_line.set_data(xs, ys)
    clock.set_text(f"t = {t[i]:.1f} s")
    return rod, bob, block, trace, spring_line, clock

# -----------------------------------------------------------
# Animation
# -----------------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n+1, STRIDE),
                     interval=STRIDE*dt*1000, blit=True)
plt.tight_layout()
plt.show()
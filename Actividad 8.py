import numpy as np
import matplotlib.pyplot as plt

# Parámetros
E = 1.0
mu = 2.8

# Frontera de la región admisible
r_max = E / (mu - 1)
r = np.linspace(0, r_max, 800)
r_dot_max = np.sqrt(np.maximum(
    0, (2 * E - 2 * r * (mu - 1)) / (mu + 1)
))
r_dot_min = -r_dot_max

fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
ax.fill_between(
    r, r_dot_min, r_dot_max,
    color="lightsteelblue", alpha=0.85,
    label="Región admisible"
)
ax.plot(r, r_dot_max, color="navy", linewidth=0.6)
ax.plot(r, r_dot_min, color="navy", linewidth=0.6)


v_max = np.sqrt(2 * E / (mu + 1))
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\dot {r}$")
ax.set_xlim(0, 0.62)
ax.set_ylim(-0.82, 0.82)
ax.grid(True, alpha=0.25)
fig.tight_layout()

# Guardar como PNG
plt.savefig("Región_angular_admisible.png", dpi=180, bbox_inches="tight")
plt.show()

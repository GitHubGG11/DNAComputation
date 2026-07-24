from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


output_directory = Path(__file__).resolve().parent
plot_data_path = output_directory / "coexistence_hull_3d.npz"
plot_path = output_directory / "coexistence_hull_3d_raw.png"

with np.load(plot_data_path) as data:
    rho_polygons = data["rho_polygons"]
    phi_polygons = data["phi_polygons"]
    delta_g_polygons = data["delta_g_polygons"]

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for rho_polygon, phi_polygon, delta_g_polygon in zip(
    rho_polygons,
    phi_polygons,
    delta_g_polygons,
):
    ax.plot(
        rho_polygon,
        phi_polygon,
        delta_g_polygon,
        color="navy",
        linewidth=1.4,
    )

ax.plot_surface(
    rho_polygons,
    phi_polygons,
    delta_g_polygons,
    color="cornflowerblue",
    alpha=0.5,
    linewidth=0,
    antialiased=True,
    shade=True,
)

ax.set_xlabel(r"Parent linker number fraction $\bar{\rho}$")
ax.set_ylabel(r"Parent nanostar number fraction $\bar{\phi}$")
ax.set_zlabel(r"$\beta\Delta G$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Parent-projected coexistence hull")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved plot: {plot_path}")
plt.show()

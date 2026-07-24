from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


volNs = 4e-21
Nav = 6.02214076e23

output_directory = Path(__file__).resolve().parent
plot_data_path = output_directory / "coexistence_hull_3d.npz"
plot_path = output_directory / "coexistence_hull_3d_replot_micromolar.png"

with np.load(plot_data_path) as data:
    rho_number_polygons = data["rho_polygons"]
    phi_number_polygons = data["phi_polygons"]
    delta_g_polygons = data["delta_g_polygons"]

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(bottom=0.27)

volume_axis = fig.add_axes([0.20, 0.14, 0.65, 0.03])
nanostar_size_axis = fig.add_axes([0.20, 0.09, 0.65, 0.03])
w_axis = fig.add_axes([0.20, 0.04, 0.65, 0.03])
volume_slider = Slider(
    volume_axis,
    r"Base volume (nm$^3$)",
    1000.0,
    10000.0,
    valinit=volNs * 10**24,
)
nanostar_size_slider = Slider(
    nanostar_size_axis,
    "Nanostar volume multiplier",
    1.0,
    10.0,
    valinit=2.0,
)
w_slider = Slider(
    w_axis,
    r"Number ratio $w$ in $\rho=w\phi$",
    0.1,
    5.0,
    valinit=2.0,
)


def draw_hull(_=None):
    ax.clear()
    base_volume_liters = volume_slider.val * 10**-24
    rho_conversion = 10**6 * nanostar_size_slider.val / (base_volume_liters * Nav)
    phi_conversion = 10**6 / (
        base_volume_liters * Nav
    )
    rho_polygons = rho_number_polygons * rho_conversion
    phi_polygons = phi_number_polygons * phi_conversion

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

    positive_phi = phi_number_polygons[phi_number_polygons > 0.0]
    phi_number_plane = np.geomspace(
        np.min(positive_phi),
        np.max(positive_phi),
        30,
    )
    delta_g_plane = np.linspace(
        np.min(delta_g_polygons),
        np.max(delta_g_polygons),
        10,
    )
    phi_number_plane, delta_g_plane = np.meshgrid(
        phi_number_plane,
        delta_g_plane,
    )
    rho_number_plane = w_slider.val * phi_number_plane
    ax.plot_surface(
        rho_number_plane * rho_conversion,
        phi_number_plane * phi_conversion,
        delta_g_plane,
        color="crimson",
        alpha=0.22,
        linewidth=0,
        shade=False,
    )

    intersection_rho = []
    intersection_phi = []
    intersection_delta_g = []
    for level in range(len(delta_g_polygons) - 1):
        for fraction in np.linspace(0.0, 1.0, 25, endpoint=False):
            rho_section = (
                (1.0 - fraction) * rho_number_polygons[level]
                + fraction * rho_number_polygons[level + 1]
            )
            phi_section = (
                (1.0 - fraction) * phi_number_polygons[level]
                + fraction * phi_number_polygons[level + 1]
            )
            delta_g_section = (
                (1.0 - fraction) * delta_g_polygons[level, 0]
                + fraction * delta_g_polygons[level + 1, 0]
            )
            difference = rho_section - w_slider.val * phi_section
            crossing_segments = np.flatnonzero(
                difference[:-1] * difference[1:] <= 0.0
            )
            for segment in crossing_segments:
                denominator = difference[segment] - difference[segment + 1]
                if abs(denominator) < 1e-14:
                    continue
                amount = difference[segment] / denominator
                intersection_rho.append(
                    rho_section[segment]
                    + amount * (rho_section[segment + 1] - rho_section[segment])
                )
                intersection_phi.append(
                    phi_section[segment]
                    + amount * (phi_section[segment + 1] - phi_section[segment])
                )
                intersection_delta_g.append(delta_g_section)

    if intersection_rho:
        ax.scatter(
            np.asarray(intersection_rho) * rho_conversion,
            np.asarray(intersection_phi) * phi_conversion,
            intersection_delta_g,
            color="red",
            s=16,
            depthshade=False,
            label="Plane-hull intersection",
        )
        ax.legend(loc="upper right")

    ax.set_xlabel(r"Parent linker concentration $\bar{\rho}$ ($\mu$M)")
    ax.set_ylabel(r"Parent nanostar concentration $\bar{\phi}$ ($\mu$M)")
    ax.set_zlabel(r"$\beta\Delta G$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        rf"Coexistence hull with number-fraction plane "
        rf"$\rho={w_slider.val:.2f}\phi$"
    )
    ax.grid(alpha=0.3)
    fig.canvas.draw_idle()


volume_slider.on_changed(draw_hull)
nanostar_size_slider.on_changed(draw_hull)
w_slider.on_changed(draw_hull)
draw_hull()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved plot: {plot_path}")
plt.show()

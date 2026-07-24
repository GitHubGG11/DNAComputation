import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.special import expit
import matplotlib.pyplot as plt

z = 4.0
EPS = 1e-8
MIN_GAP = 1e-4
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
PLOT_PATH = OUTPUT_DIRECTORY / "coexistence_hull_3d.png"
PLOT_DATA_PATH = OUTPUT_DIRECTORY / "coexistence_hull_3d.npz"


def h(y, G, phi):
    k = np.exp(-G)

    u = y * k / (1.0 + y)
    q = y * k**2 / ((1.0 + y) * (1.0 + u)**2)

    return (
        phi * np.log(phi)
        + (1.0 - phi) * np.log(1.0 - phi)
        - (1.0 - phi) * np.log1p(y)
        - z * phi * (1.0 - phi) * np.log1p(u)
        - (z / 2.0) * phi**2 * (1.0 - phi) * np.log1p(q)
    )


def dh(y, G, phi):
    k = np.exp(-G)

    u = y * k / (1.0 + y)
    q = y * k**2 / ((1.0 + y) * (1.0 + u)**2)

    return (
        np.log(phi / (1.0 - phi))
        + np.log1p(y)
        - z * (1.0 - 2.0 * phi) * np.log1p(u)
        - (z / 2.0) * (2.0 * phi - 3.0 * phi**2) * np.log1p(q)
    )


def ddh(y, G, phi):
    k = np.exp(-G)

    u = y * k / (1.0 + y)
    q = y * k**2 / ((1.0 + y) * (1.0 + u)**2)

    return (
        1.0 / (phi * (1.0 - phi))
        + 2.0 * z * np.log1p(u)
        + z * (3.0 * phi - 1.0) * np.log1p(q)
    )


def unpack_vars(vars_):
    """
    Transform unconstrained variables into compositions satisfying

        EPS < phi_minus
        phi_minus + MIN_GAP < phi_plus
        phi_plus < 1 - EPS
    """
    a, b = vars_

    phi_minus = (
        EPS
        + (1.0 - 2.0 * EPS - MIN_GAP) * expit(a)
    )

    remaining = 1.0 - EPS - phi_minus - MIN_GAP

    phi_plus = (
        phi_minus
        + MIN_GAP
        + remaining * expit(b)
    )

    return phi_minus, phi_plus


def tangent_residual(vars_, y, G):
    phi_minus, phi_plus = unpack_vars(vars_)

    slope_minus = dh(y, G, phi_minus)
    slope_plus = dh(y, G, phi_plus)

    eq1 = slope_minus - slope_plus

    eq2 = (
        h(y, G, phi_plus)
        - h(y, G, phi_minus)
        - slope_minus * (phi_plus - phi_minus)
    )

    return np.array([eq1, eq2])


def rho(y, G, phi):

    k = np.exp(-G)

    u = y * k / (1.0 + y)
    q = y * k**2 / ((1.0 + y) * (1.0 + u)**2)

    # Numbers of available single- and double-binding configurations per site
    ns_bar = z * phi * (1.0 - phi)
    nd_bar = z /2 * phi**2 * (1.0 - phi)

    # Linker densities per lattice site
    d_bar = nd_bar * q / (1.0 + q)
    s_bar = (ns_bar - 2.0 * d_bar) * u / (1.0 + u)
    l_bar = (y / (1.0 + y)) * (
        1.0 - phi - s_bar - d_bar
    )

    rho_bar = l_bar + s_bar + d_bar

    return rho_bar

betaG_values = np.linspace(-4.0, -15.0, 50)

# Multiple starting points because this is a nonlinear problem.
initial_guesses = [
    (-4.0, -2.0),
    (-4.0, 0.0),
    (-4.0, 2.0),
    (-2.0, -2.0),
    (-2.0, 0.0),
    (-2.0, 2.0),
    (0.0, -2.0),
    (0.0, 0.0),
    (0.0, 2.0),
]


def solve_y_point(task):
    y, betaG = task
    phi_grid = np.linspace(1e-5, 1.0 - 1e-5, 4000)
    if np.min(ddh(y, betaG, phi_grid)) >= 0.0:
        return None

    best_solution = None
    for x0 in initial_guesses:
        solution = least_squares(
            tangent_residual,
            x0=x0,
            args=(y, betaG),
            max_nfev=3000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        phi_minus, phi_plus = unpack_vars(solution.x)
        residual_norm = np.linalg.norm(solution.fun)
        if best_solution is None or residual_norm < best_solution[0]:
            best_solution = residual_norm, phi_minus, phi_plus

    residual_norm, phi_minus, phi_plus = best_solution
    if residual_norm >= 1e-8:
        return None
    return (
        y,
        phi_minus,
        phi_plus,
        rho(y, betaG, phi_minus),
        rho(y, betaG, phi_plus),
    )


plot_results = []

for betaG in betaG_values:
    # Store all reliable tie lines from the y sweep.
    tie_lines = []
    y_sweep = 10.0 ** (-np.linspace(3, 9, 100))

    with ThreadPoolExecutor() as executor:
        solved_points = executor.map(
            solve_y_point,
            [(y, betaG) for y in y_sweep],
        )
        for sweep_index, result in enumerate(solved_points, 1):
            if result is not None:
                tie_lines.append(result)
            filled = 30 * sweep_index // len(y_sweep)
            print(
                f"\rDelta G = {betaG:6.1f} "
                f"[{'#' * filled}{'-' * (30 - filled)}] "
                f"{sweep_index}/{len(y_sweep)}",
                end="",
                flush=True,
            )

    print()

        # print(
        #     f"y = {y:.1e}: "
        #     f"phi_minus = {phi_minus:.8f}, "
        #     f"phi_plus = {phi_plus:.8f}, "
        #     f"residual = {residual_norm:.3e}, "
        #     f"rho_minus = {rho_minus:.8f}, "
        #     f"rho_plus = {rho_plus:.8f}"
        # )
    # else:
    #     print(
    #         f"y = {y:.1e}: nonconvex curvature found, "
    #         "but no reliable common tangent was located. "
    #         f"Best residual = {residual_norm:.3e}"
    #     )


# ============================================================
# Parent-projected coexistence region
# ============================================================

    if len(tie_lines) == 0:
        print(f"No valid tie lines found for betaDeltaG={betaG}")
        continue

# Full rho range covered by at least one tie line.
    rho_bar_min = min(
        min(rho_minus, rho_plus)
        for _, _, _, rho_minus, rho_plus in tie_lines
    )

    rho_bar_max = max(
        max(rho_minus, rho_plus)
        for _, _, _, rho_minus, rho_plus in tie_lines
    )

# Parent linker densities at which to determine the allowed phi range.
    rho_bar_grid = np.linspace(
        rho_bar_min,
        rho_bar_max,
        400,
    )

    phi_bar_min = np.full_like(rho_bar_grid, np.nan)
    phi_bar_max = np.full_like(rho_bar_grid, np.nan)

    for j, rho_bar in enumerate(rho_bar_grid):
        candidate_phi_bars = []

        for (
            y,
            phi_minus,
            phi_plus,
            rho_minus,
            rho_plus,
        ) in tie_lines:

            delta_rho = rho_plus - rho_minus

        # Degenerate tie line in the rho direction.
            if abs(delta_rho) < 1e-12:
                continue

        # Fraction of the plus phase required by linker conservation.
            lambda_plus = (
                rho_bar - rho_minus
            ) / delta_rho

        # rho_bar lies on this tie line only when 0 <= lambda <= 1.
            if -1e-10 <= lambda_plus <= 1.0 + 1e-10:
                lambda_plus = np.clip(
                    lambda_plus,
                    0.0,
                    1.0,
                )

            # Nanostar lever rule.
                phi_bar = (
                    phi_minus
                    + lambda_plus
                    * (phi_plus - phi_minus)
                )

                candidate_phi_bars.append(phi_bar)

        if candidate_phi_bars:
            phi_bar_min[j] = np.min(candidate_phi_bars)
            phi_bar_max[j] = np.max(candidate_phi_bars)


# Remove rho values not crossed by any tie line.
    valid = (
        np.isfinite(phi_bar_min)
        & np.isfinite(phi_bar_max)
    )

    rho_bar_plot = rho_bar_grid[valid]
    phi_bar_min_plot = phi_bar_min[valid]
    phi_bar_max_plot = phi_bar_max[valid]
    plot_results.append(
        (betaG, rho_bar_plot, phi_bar_min_plot, phi_bar_max_plot)
    )


# Plot the parent-projected coexistence hull in 3D.
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

rho_polygons = []
phi_polygons = []
delta_g_polygons = []

for betaG, rho_plot, phi_min_plot, phi_max_plot in plot_results:
    rho_smooth = np.linspace(rho_plot[0], rho_plot[-1], 300)
    phi_min_smooth = PchipInterpolator(rho_plot, phi_min_plot)(rho_smooth)
    phi_max_smooth = PchipInterpolator(rho_plot, phi_max_plot)(rho_smooth)
    rho_polygon = np.concatenate(
        [rho_smooth, rho_smooth[::-1], rho_smooth[:1]]
    )
    phi_polygon = np.concatenate(
        [phi_min_smooth, phi_max_smooth[::-1], phi_min_smooth[:1]]
    )

    rho_polygons.append(rho_polygon)
    phi_polygons.append(phi_polygon)
    delta_g_polygons.append(np.full_like(rho_polygon, betaG))
    ax.plot(
        rho_polygon,
        phi_polygon,
        betaG,
        color="navy",
        linewidth=1.4,
    )

ax.plot_surface(
    np.asarray(rho_polygons),
    np.asarray(phi_polygons),
    np.asarray(delta_g_polygons),
    color="cornflowerblue",
    alpha=0.5,
    linewidth=0,
    antialiased=True,
    shade=True,
)

ax.set_xlabel(r"Parent linker density $\bar{\rho}$")
ax.set_ylabel(r"Parent nanostar fraction $\bar{\phi}$")
ax.set_zlabel(r"$\beta\Delta G$")
ax.set_title("Parent-projected coexistence hull")
ax.grid(alpha=0.3)
plt.tight_layout()
np.savez_compressed(
    PLOT_DATA_PATH,
    rho_polygons=np.asarray(rho_polygons),
    phi_polygons=np.asarray(phi_polygons),
    delta_g_polygons=np.asarray(delta_g_polygons),
)
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved plot: {PLOT_PATH}")
print(f"Saved plot data: {PLOT_DATA_PATH}")
plt.show()

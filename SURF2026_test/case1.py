from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy


# ======================================================================
# User parameters
# ======================================================================

PHI_BAR = 0.001
RHO_BAR = 0.002

COORDINATION_Z = 4.0

# In the manuscript:
#
#   n_d / N = z * z_star * phi^2 * (1 - phi)
#
# Set z_star = 1/2 to obtain the coefficient z/2.
Z_STAR = 0.5

BETA_DELTA_G_MIN = -14.0
BETA_DELTA_G_MAX = -20.0
NUMBER_OF_ENERGY_VALUES = 10

LOG_Y_MIN = -100.0
LOG_Y_MAX = 8.0
NUMBER_OF_Y_VALUES = 400

RHO_MATCH_TOLERANCE = 5.0e-4


PHI_GRID = np.unique(
    np.concatenate(
        [
            np.geomspace(1.0e-7, 0.02, 400),
            np.linspace(0.02, 0.60, 700),
            np.linspace(0.60, 0.999, 300),
        ]
    )
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent

CSV_PATH = OUTPUT_DIRECTORY / "binodal_case1_only.csv"
PLOT_PATH = OUTPUT_DIRECTORY / "binodal_case1_only.png"


# ======================================================================
# Thermodynamic functions
# ======================================================================

def mixing_free_energy(phi: np.ndarray) -> np.ndarray:
    """
    Ideal lattice mixing contribution:

        phi ln(phi) + (1 - phi) ln(1 - phi)
    """
    solvent = 1.0 - phi

    return (
        xlogy(phi, phi)
        + xlogy(solvent, solvent)
    )


def case1_homogeneous(
    phi: np.ndarray,
    y: float,
    K: float,
    z: float,
    z_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the exact Case 1 semi-grand free energy and linker density.

    Definitions:

        u = yK / (1 + y)

        q = [yK^2 / (1 + y)] / (1 + u)^2

        n_s / N = z phi (1 - phi)

        n_d / N = z z_star phi^2 (1 - phi)

    Case 1 stationary populations:

        d_0 / N = (n_d / N) q / (1 + q)

        s_0 / N = [(n_s / N) - 2d_0/N] u / (1 + u)

    The function returns:

        h       semi-grand free-energy density
        rho     total linker density
        valid   whether the Case 1 stationary point is feasible
    """
    solvent = 1.0 - phi

    n_s = z * phi * solvent
    n_d = z / 2 * phi**2 * solvent

    u = y * K / (1.0 + y)

    q = (
        y * K**2 / (1.0 + y)
    ) / (1.0 + u) ** 2

    d = n_d * q / (1.0 + q)

    available_single_sites = n_s - 2.0 * d

    s = available_single_sites * u / (1.0 + u)

    free_linker_sites = solvent - s - d

    h = (
        mixing_free_energy(phi)
        - solvent * np.log1p(y)
        - n_s * np.log1p(u)
        - n_d * np.log1p(q)
    )

    h_approx = (
        mixing_free_energy(phi)
        - solvent * np.log1p(y)
        - n_s * np.log1p(u)
        - n_d * np.log1p(q)
    )


    # Free linkers occupy the remaining solvent sites with probability
    # y / (1 + y).
    free_linkers = (
        y / (1.0 + y)
    ) * free_linker_sites

    rho = s + d + free_linkers

    valid = (
        (available_single_sites >= -1.0e-12)
        & (s >= -1.0e-12)
        & (d >= -1.0e-12)
        & (s + d <= solvent + 1.0e-12)
        & (d <= n_d + 1.0e-12)
        & (2.0 * d <= n_s + 1.0e-12)
    )

    h = np.where(valid, h, np.inf)
    rho = np.where(valid, rho, np.nan)

    return h, rho, valid


# ======================================================================
# Convex-hull construction
# ======================================================================

def lower_convex_hull_indices(
    phi: np.ndarray,
    h: np.ndarray,
) -> list[int]:
    """
    Return the indices on the lower convex hull of h(phi).

    A nonadjacent hull segment corresponds to a two-phase coexistence
    tie line.
    """
    hull: list[int] = []

    for index in range(phi.size):
        if not np.isfinite(h[index]):
            continue

        while len(hull) >= 2:
            first = hull[-2]
            second = hull[-1]

            slope_12 = (
                h[second] - h[first]
            ) / (
                phi[second] - phi[first]
            )

            slope_23 = (
                h[index] - h[second]
            ) / (
                phi[index] - phi[second]
            )

            if slope_12 >= slope_23 - 1.0e-12:
                hull.pop()
            else:
                break

        hull.append(index)

    return hull


@dataclass
class TieLine:
    log_y: float
    y: float

    phi_minus: float
    phi_plus: float

    rho_minus: float
    rho_plus: float

    lambda_plus: float
    rho_parent_predicted: float


def find_tie_line(
    phi: np.ndarray,
    h: np.ndarray,
    rho: np.ndarray,
    phi_bar: float,
    log_y: float,
) -> Optional[TieLine]:
    """
    Find the Case 1 coexistence segment containing phi_bar.

    The nanostar lever rule is

        phi_bar = (1-lambda) phi_minus + lambda phi_plus.

    This determines lambda. The corresponding predicted parent linker
    density is

        rho_predicted
            = (1-lambda) rho_minus + lambda rho_plus.
    """
    hull = lower_convex_hull_indices(phi, h)

    possible_tie_lines: list[TieLine] = []

    for left, right in zip(hull[:-1], hull[1:]):
        # Adjacent points merely trace the homogeneous convex region.
        if right <= left + 1:
            continue

        phi_minus = phi[left]
        phi_plus = phi[right]

        if not (
            phi_minus <= phi_bar <= phi_plus
        ):
            continue

        if not (
            np.isfinite(rho[left])
            and np.isfinite(rho[right])
        ):
            continue

        lambda_plus = (
            phi_bar - phi_minus
        ) / (
            phi_plus - phi_minus
        )

        rho_parent_predicted = (
            (1.0 - lambda_plus) * rho[left]
            + lambda_plus * rho[right]
        )

        possible_tie_lines.append(
            TieLine(
                log_y=log_y,
                y=float(np.exp(log_y)),
                phi_minus=float(phi_minus),
                phi_plus=float(phi_plus),
                rho_minus=float(rho[left]),
                rho_plus=float(rho[right]),
                lambda_plus=float(lambda_plus),
                rho_parent_predicted=float(
                    rho_parent_predicted
                ),
            )
        )

    if not possible_tie_lines:
        return None

    # Usually there should be only one relevant tie line. If numerical
    # discretization produces several, keep the widest coexistence interval.
    return max(
        possible_tie_lines,
        key=lambda tie: tie.phi_plus - tie.phi_minus,
    )


def tie_line_at_log_y(
    beta_delta_g: float,
    log_y: float,
) -> Optional[TieLine]:
    """
    Construct the Case 1 tie line for a specified beta Delta G and log(y).
    """
    K = float(np.exp(-beta_delta_g))
    y = float(np.exp(log_y))

    h, rho, _ = case1_homogeneous(
        phi=PHI_GRID,
        y=y,
        K=K,
        z=COORDINATION_Z,
        z_star=Z_STAR,
    )

    return find_tie_line(
        phi=PHI_GRID,
        h=h,
        rho=rho,
        phi_bar=PHI_BAR,
        log_y=log_y,
    )


# ======================================================================
# Activity matching
# ======================================================================

def solve_beta_delta_g(
    beta_delta_g: float,
) -> Optional[dict[str, float]]:
    """
    Find y such that the Case 1 coexistence tie line satisfies both parent
    lever rules:

        phi_bar
            = (1-lambda) phi_minus + lambda phi_plus

        rho_bar
            = (1-lambda) rho_minus + lambda rho_plus.
    """
    log_y_values = np.linspace(
        LOG_Y_MIN,
        LOG_Y_MAX,
        NUMBER_OF_Y_VALUES,
    )

    samples: list[
        Optional[tuple[float, TieLine]]
    ] = []

    for log_y in log_y_values:
        tie = tie_line_at_log_y(
            beta_delta_g=beta_delta_g,
            log_y=float(log_y),
        )

        if tie is None:
            samples.append(None)
            continue

        residual = (
            tie.rho_parent_predicted - RHO_BAR
        )

        samples.append((residual, tie))

    roots: list[TieLine] = []

    for index in range(len(log_y_values) - 1):
        left_sample = samples[index]
        right_sample = samples[index + 1]

        # Do not bridge an interval in which no valid Case 1 tie line exists.
        if left_sample is None or right_sample is None:
            continue

        left_residual, left_tie = left_sample
        right_residual, right_tie = right_sample

        if abs(left_residual) <= 1.0e-12:
            roots.append(left_tie)
            continue

        if left_residual * right_residual > 0.0:
            continue

        left_log_y = float(log_y_values[index])
        right_log_y = float(log_y_values[index + 1])

        best_tie = min(
            [left_tie, right_tie],
            key=lambda tie: abs(
                tie.rho_parent_predicted - RHO_BAR
            ),
        )

        # Bisection in log(y).
        for _ in range(60):
            midpoint_log_y = 0.5 * (
                left_log_y + right_log_y
            )

            midpoint_tie = tie_line_at_log_y(
                beta_delta_g=beta_delta_g,
                log_y=midpoint_log_y,
            )

            if midpoint_tie is None:
                break

            midpoint_residual = (
                midpoint_tie.rho_parent_predicted
                - RHO_BAR
            )

            if abs(midpoint_residual) < abs(
                best_tie.rho_parent_predicted
                - RHO_BAR
            ):
                best_tie = midpoint_tie

            if abs(midpoint_residual) <= 1.0e-10:
                break

            if (
                left_residual * midpoint_residual
                <= 0.0
            ):
                right_log_y = midpoint_log_y
                right_residual = midpoint_residual
            else:
                left_log_y = midpoint_log_y
                left_residual = midpoint_residual

        roots.append(best_tie)

    if not roots:
        return None

    best = min(
        roots,
        key=lambda tie: abs(
            tie.rho_parent_predicted - RHO_BAR
        ),
    )

    rho_residual = abs(
        best.rho_parent_predicted - RHO_BAR
    )

    if rho_residual > RHO_MATCH_TOLERANCE:
        return None

    K = float(np.exp(-beta_delta_g))

    return {
        "beta_Delta_G": beta_delta_g,
        "K": K,
        "y_star": best.y,
        "log_y_star": best.log_y,
        "phi_minus": best.phi_minus,
        "phi_plus": best.phi_plus,
        "rho_minus": best.rho_minus,
        "rho_plus": best.rho_plus,
        "lambda_plus": best.lambda_plus,
        "rho_parent_predicted": (
            best.rho_parent_predicted
        ),
        "rho_residual": rho_residual,
    }


# ======================================================================
# Main sweep
# ======================================================================

def main() -> None:
    rows: list[dict[str, float]] = []

    beta_delta_g_values = np.linspace(
        BETA_DELTA_G_MIN,
        BETA_DELTA_G_MAX,
        NUMBER_OF_ENERGY_VALUES,
    )

    for beta_delta_g in beta_delta_g_values:
        row = solve_beta_delta_g(
            float(beta_delta_g)
        )

        if row is None:
            print(
                f"beta Delta G = {beta_delta_g:8.3f}: "
                "no parent-compatible Case 1 tie line"
            )
            continue

        ystar = row['y_star']
        K = np.exp(-beta_delta_g)
        ustar = ystar * K / (1.0 + ystar)

        qstar = (
            ystar * K**2 / (1.0 + ystar)
        ) / (1.0 + ustar) ** 2

        print(ustar, ystar * K, np.log1p(ustar), np.log1p(qstar), np.log((K) / (2 + ystar*K)))

        rows.append(row)

        print(
            f"beta Delta G = {beta_delta_g:8.3f}, "
            f"y* = {row['y_star']:.6e}, "
            f"phi- = {row['phi_minus']:.6f}, "
            f"phi+ = {row['phi_plus']:.6f}, "
            f"rho- = {row['rho_minus']:.6f}, "
            f"rho+ = {row['rho_plus']:.6f}, "
            f"rho error = {row['rho_residual']:.3e}"
        )

    if not rows:
        raise RuntimeError(
            "No parent-compatible Case 1 binodal points were found."
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Plot binodal
    # ------------------------------------------------------------------

    beta_delta_g = np.array(
        [row["beta_Delta_G"] for row in rows]
    )

    phi_minus = np.array(
        [row["phi_minus"] for row in rows]
    )

    phi_plus = np.array(
        [row["phi_plus"] for row in rows]
    )

    plt.figure(figsize=(8.2, 5.4))

    plt.plot(
        phi_minus,
        beta_delta_g,
        marker="o",
        markersize=3,
        label=r"Dilute phase $\phi_-$",
    )

    plt.plot(
        phi_plus,
        beta_delta_g,
        marker="o",
        markersize=3,
        label=r"Dense phase $\phi_+$",
    )

    plt.axvline(
        PHI_BAR,
        linestyle="--",
        linewidth=1,
        label=r"Parent $\bar{\phi}$",
    )

    plt.xlabel(
        r"Coexisting nanostar fraction $\phi$"
    )

    plt.ylabel(
        r"$\beta\Delta G$"
    )

    plt.title(
        rf"Case 1 parent-projected binodal: "
        rf"$\bar{{\phi}}={PHI_BAR}$, "
        rf"$\bar{{\rho}}={RHO_BAR}$"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=220)
    plt.close()

    print()
    print(f"Saved CSV:  {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    print(
        "Maximum accepted rho residual: "
        f"{max(row['rho_residual'] for row in rows):.3e}"
    )


if __name__ == "__main__":
    main()
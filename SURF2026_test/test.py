"""
Checked parent-projected binodal for the linker lattice model.
One sticky end for beta G. 
The plotted control parameter is beta*DeltaG. Internally, K=exp(-beta*DeltaG).
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy


# ---------------------------- User parameters ---------------------------- #

PHI_BAR = 0.004
RHO_BAR = 0.006

COORDINATION_Z = 4.0
Z_STAR = 1

BETA_DELTA_G_MIN = -1
BETA_DELTA_G_MAX = -15
NUMBER_OF_ENERGY_VALUES = 10

# Search a broad activity interval. Do not center the search at -2 beta_G:
# for finite parent linker density, the correct root can be much larger.
LOG_Y_MIN = -40.0
LOG_Y_MAX = 8.0
NUMBER_OF_Y_VALUES = 240
RHO_MATCH_TOLERANCE = 5.0e-4

PHI_GRID = np.unique(
    np.concatenate(
        [
            np.geomspace(1.0e-7, 0.02, 320),
            np.linspace(0.02, 0.60, 560),
            np.linspace(0.60, 0.95, 120),
        ]
    )
)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIRECTORY / "binodal_results_betaDeltaG_corrected.csv"
PLOT_PATH = OUTPUT_DIRECTORY / "binodal_phi_x_betaDeltaG_corrected.png"


def B(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """B(A,b)=A ln A-b ln b-(A-b)ln(A-b)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    r = A - b
    result = xlogy(A, A) - xlogy(b, b) - xlogy(r, r)
    invalid = (A < -1e-11) | (b < -1e-11) | (r < -1e-11)
    return np.where(invalid, np.nan, result)


def lower_convex_hull_indices(phi: np.ndarray, h: np.ndarray) -> list[int]:
    hull: list[int] = []
    for i in range(phi.size):
        if not np.isfinite(h[i]):
            continue
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            slope_12 = (h[i2] - h[i1]) / (phi[i2] - phi[i1])
            slope_23 = (h[i] - h[i2]) / (phi[i] - phi[i2])
            if slope_12 >= slope_23 - 1e-12:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def solve_case2_x_vectorized(
    phi: np.ndarray, K: float, z: float, z_star: float
) -> np.ndarray:
    """
    Solve
      K(A-x)(1-x)(z phi-1-x)=x(z phi-2x)^2
    with A=z z_star phi^2 and x=d/[N(1-phi)].
    """
    A = z * Z_STAR * 1/2 * phi * phi
    zphi = z * phi
    upper = np.minimum.reduce(
        [A, np.ones_like(phi), zphi - 1.0, zphi / 2.0]
    )
    valid = upper >= -1e-13
    x = np.full_like(phi, np.nan)

    tiny = valid & (upper <= 1e-13) & (zphi >= 1.0 - 1e-12)
    x[tiny] = 0.0

    active = valid & (upper > 1e-13)
    if not np.any(active):
        return x

    Aa = A[active]
    za = zphi[active]
    hi = upper[active]
    lo = np.zeros_like(hi)

    def F(value: np.ndarray) -> np.ndarray:
        return (
            K * (Aa - value) * (1.0 - value) * (za - 1.0 - value)
            - value * (za - 2.0 * value) ** 2
        )

    f_lo = F(lo)
    f_hi = F(hi)

    at_lower = f_lo <= 0.0
    at_upper = f_hi >= 0.0
    interior = ~(at_lower | at_upper)

    answer = np.empty_like(hi)
    answer[at_lower] = 0.0
    answer[at_upper] = hi[at_upper]

    left = lo[interior]
    right = hi[interior]
    A_i = Aa[interior]
    z_i = za[interior]

    for _ in range(58):
        mid = 0.5 * (left + right)
        f_mid = (
            K * (A_i - mid) * (1.0 - mid) * (z_i - 1.0 - mid)
            - mid * (z_i - 2.0 * mid) ** 2
        )
        move_left = f_mid > 0.0
        left = np.where(move_left, mid, left)
        right = np.where(move_left, right, mid)

    answer[interior] = 0.5 * (left + right)
    x[active] = answer
    return x


@dataclass
class KData:
    phi: np.ndarray
    c: np.ndarray
    a_s: np.ndarray
    a_d: np.ndarray
    mixing: np.ndarray
    x2: np.ndarray
    h2_constant: np.ndarray
    d3: np.ndarray
    valid3: np.ndarray
    h3_constant: np.ndarray


def precompute_K(phi: np.ndarray, K: float, z: float, z_star: float) -> KData:
    c = 1.0 - phi
    a_s = z * phi * c
    a_d = z * Z_STAR * 1/2 * phi * phi * c
    mixing = xlogy(phi, phi) + xlogy(c, c)

    x2 = solve_case2_x_vectorized(phi, K, z, z_star)
    h2_constant = np.full_like(phi, np.inf)
    valid2 = np.isfinite(x2)
    if np.any(valid2):
        p = phi[valid2]
        cc = c[valid2]
        xx = x2[valid2]
        h2_constant[valid2] = mixing[valid2] - cc * (
            B(z * Z_STAR * 1/2 * p * p, xx)
            + B(z * p - 2.0 * xx, 1.0 - xx)
            + (1.0 + xx) * np.log(K)
        )

    d3 = a_s / 2.0
    valid3 = (d3 <= a_d + 1e-12) & (d3 <= c + 1e-12)
    h3_constant = np.full_like(phi, np.inf)
    h3_constant[valid3] = (
        mixing[valid3]
        - B(a_d[valid3], d3[valid3])
        - 2.0 * d3[valid3] * np.log(K)
    )

    return KData(
        phi=phi,
        c=c,
        a_s=a_s,
        a_d=a_d,
        mixing=mixing,
        x2=x2,
        h2_constant=h2_constant,
        d3=d3,
        valid3=valid3,
        h3_constant=h3_constant,
    )


def homogeneous(pre: KData, y: float, K: float):
    """
    Candidate cases:
      1: unconstrained interior
      2: s+d=1-phi
      3: a_s-2d=0
    """
    c, a_s, a_d = pre.c, pre.a_s, pre.a_d

    u = y * K / (1.0 + y)
    q = (y * K * K / (1.0 + y)) / (1.0 + u) ** 2

    d0 = a_d * q / (1.0 + q)
    s0 = (a_s - 2.0 * d0) * u / (1.0 + u)

    h1 = (
        pre.mixing
        - c * np.log1p(y)
        - a_s * np.log1p(u)
        - a_d * np.log1p(q)
    )
    rho1 = d0 + s0 + y / (1.0 + y) * (c - d0 - s0)
    valid1 = (
        (d0 >= -1e-12)
        & (s0 >= -1e-12)
        & (d0 <= a_s / 2.0 + 1e-12)
        & (d0 + s0 <= c + 1e-12)
    )

    n = pre.phi.size
    h_candidates = np.full((3, n), np.inf)
    rho_candidates = np.full((3, n), np.nan)

    h_candidates[0, valid1] = h1[valid1]
    rho_candidates[0, valid1] = rho1[valid1]

    valid2 = np.isfinite(pre.x2)
    h_candidates[1, valid2] = (
        pre.h2_constant[valid2] - c[valid2] * np.log(y)
    )
    rho_candidates[1, valid2] = c[valid2]

    valid3 = pre.valid3
    h_candidates[2, valid3] = (
        pre.h3_constant[valid3]
        - pre.d3[valid3] * np.log(y)
        - (c[valid3] - pre.d3[valid3]) * np.log1p(y)
    )
    rho_candidates[2, valid3] = (
        pre.d3[valid3]
        + y / (1.0 + y) * (c[valid3] - pre.d3[valid3])
    )

    selected = np.argmin(h_candidates, axis=0)
    indices = np.arange(n)
    return (
        h_candidates[selected, indices],
        rho_candidates[selected, indices],
        selected + 1,
    )


@dataclass
class TieLine:
    log_y: float
    y: float
    phi_minus: float
    phi_plus: float
    rho_minus: float
    rho_plus: float
    lambda_plus: float
    rho_mix: float
    case_minus: int
    case_plus: int


def tie_line(pre: KData, K: float, log_y: float, phi_bar: float) -> Optional[TieLine]:
    y = float(np.exp(log_y))
    h, rho, case = homogeneous(pre, y, K)
    hull = lower_convex_hull_indices(pre.phi, h)

    found = []
    for i, j in zip(hull[:-1], hull[1:]):
        if j <= i + 1:
            continue
        if pre.phi[i] <= phi_bar <= pre.phi[j]:
            lam = (phi_bar - pre.phi[i]) / (pre.phi[j] - pre.phi[i])
            rho_mix = (1.0 - lam) * rho[i] + lam * rho[j]
            found.append(
                TieLine(
                    log_y=log_y,
                    y=y,
                    phi_minus=float(pre.phi[i]),
                    phi_plus=float(pre.phi[j]),
                    rho_minus=float(rho[i]),
                    rho_plus=float(rho[j]),
                    lambda_plus=float(lam),
                    rho_mix=float(rho_mix),
                    case_minus=int(case[i]),
                    case_plus=int(case[j]),
                )
            )

    if not found:
        return None
    return max(found, key=lambda item: item.phi_plus - item.phi_minus)



def solve_beta_delta_G(beta_delta_G: float) -> Optional[dict]:
    """
    Find the activity y whose coexistence tie line satisfies both parent
    lever rules. Unlike the previous version, this function never accepts a
    merely 'closest' y when the linker-density residual does not cross zero.
    """
    beta_G = -beta_delta_G
    K = float(np.exp(beta_G))
    pre = precompute_K(PHI_GRID, K, COORDINATION_Z, Z_STAR)

    log_y_grid = np.linspace(LOG_Y_MIN, LOG_Y_MAX, NUMBER_OF_Y_VALUES)

    # Keep the original grid adjacency. Two valid samples separated by an
    # interval with no tie line must not be treated as a root bracket.
    sampled = []
    for log_y in log_y_grid:
        result = tie_line(pre, K, float(log_y), PHI_BAR)
        if result is None:
            sampled.append(None)
        else:
            sampled.append((result.rho_mix - RHO_BAR, result))

    roots = []

    for index in range(len(log_y_grid) - 1):
        left_sample = sampled[index]
        right_sample = sampled[index + 1]

        if left_sample is None or right_sample is None:
            continue

        left_residual, left_tie = left_sample
        right_residual, right_tie = right_sample

        if left_residual == 0.0:
            roots.append(left_tie)
            continue

        if left_residual * right_residual >= 0.0:
            continue

        left_log_y = float(log_y_grid[index])
        right_log_y = float(log_y_grid[index + 1])
        best_tie = min(
            (left_tie, right_tie),
            key=lambda value: abs(value.rho_mix - RHO_BAR),
        )

        # Bisection in log y. The finite phi grid makes the residual mildly
        # piecewise constant, so retain the best point encountered.
        for _ in range(45):
            midpoint = 0.5 * (left_log_y + right_log_y)
            midpoint_tie = tie_line(pre, K, midpoint, PHI_BAR)

            if midpoint_tie is None:
                break

            midpoint_residual = midpoint_tie.rho_mix - RHO_BAR

            if abs(midpoint_residual) < abs(best_tie.rho_mix - RHO_BAR):
                best_tie = midpoint_tie

            if abs(midpoint_residual) <= 1.0e-10:
                break

            if left_residual * midpoint_residual <= 0.0:
                right_log_y = midpoint
                right_residual = midpoint_residual
            else:
                left_log_y = midpoint
                left_residual = midpoint_residual

        roots.append(best_tie)

    if not roots:
        return None

    best = min(roots, key=lambda value: abs(value.rho_mix - RHO_BAR))
    residual = abs(best.rho_mix - RHO_BAR)

    # Never draw a point that does not actually reproduce rho_bar.
    if residual > RHO_MATCH_TOLERANCE:
        return None

    return {
        "beta_Delta_G": beta_delta_G,
        "beta_G": beta_G,
        "K": K,
        "y_star": best.y,
        "log_y_star": best.log_y,
        "phi_minus": best.phi_minus,
        "phi_plus": best.phi_plus,
        "rho_minus": best.rho_minus,
        "rho_plus": best.rho_plus,
        "lambda_plus": best.lambda_plus,
        "rho_parent_predicted": best.rho_mix,
        "rho_residual": residual,
        "case_minus": best.case_minus,
        "case_plus": best.case_plus,
    }


def main() -> None:
    rows = []

    beta_delta_G_values = np.linspace(
        BETA_DELTA_G_MIN,
        BETA_DELTA_G_MAX,
        NUMBER_OF_ENERGY_VALUES,
    )

    for beta_delta_G in beta_delta_G_values:
        row = solve_beta_delta_G(float(beta_delta_G))

        if row is None:
            print(
                f"beta Delta G={beta_delta_G:.3f}: "
                "no parent-compatible lever-rule root"
            )
            continue

        rows.append(row)

        print(
            f"beta Delta G={beta_delta_G:7.3f}, "
            f"y*={row['y_star']:.5e}, "
            f"phi-={row['phi_minus']:.6f} [Case {row['case_minus']}], "
            f"phi+={row['phi_plus']:.6f} [Case {row['case_plus']}], "
            f"rho error={row['rho_residual']:.3e}"
        )

        if row["case_minus"] == 2 or row["case_plus"] == 2:
            print("  >>> CASE 2 IS USED BY A COEXISTING PHASE.")

    if not rows:
        raise RuntimeError("No parent-compatible tie lines were found.")

    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    beta_delta_G = np.array([row["beta_Delta_G"] for row in rows])
    dilute = np.array([row["phi_minus"] for row in rows])
    dense = np.array([row["phi_plus"] for row in rows])

    plt.figure(figsize=(8.2, 5.4))
    plt.plot(
        dilute,
        beta_delta_G,
        marker="o",
        markersize=3,
        label=r"Dilute phase $\phi_-$",
    )
    plt.plot(
        dense,
        beta_delta_G,
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
    plt.xlabel(r"Coexisting nanostar fraction $\phi$")
    plt.ylabel(r"$\beta\Delta G$")
    plt.title(
        rf"Parent-projected binodal: "
        rf"$\bar{{\phi}}={PHI_BAR}$, $\bar{{\rho}}={RHO_BAR}$"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=220)
    plt.close()

    selected_cases = {
        int(row["case_minus"]) for row in rows
    } | {
        int(row["case_plus"]) for row in rows
    }

    print()
    print(f"Saved CSV:  {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Selected coexistence cases: {sorted(selected_cases)}")
    print(f"Case 2 selected: {2 in selected_cases}")
    print(
        "Maximum accepted lever-rule rho residual: "
        f"{max(row['rho_residual'] for row in rows):.3e}"
    )


if __name__ == "__main__":
    main()
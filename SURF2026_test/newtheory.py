#!/usr/bin/env python3
"""
Two-phase coexistence with ALL linkers included in total concentration.

Definitions
-----------
phi : total A-particle volume fraction
ell : total linker volume fraction, including free and bound linkers
s   : solvent fraction = 1 - phi - ell
m   : concentration of complete linker bridges

Therefore:
    phi + ell + s = 1
    ell = ell_free + m
    ell_free = ell - m

Each complete bridge:
    - consumes one linker,
    - consumes two sticky arms,
    - has binding free-energy gain epsilon_bridge.

For four sticky arms per A particle:
    total arm concentration = 4 phi
    maximum bridge concentration = 2 phi

The homogeneous free energy is minimized over m, and the code then finds
the globally optimal two-phase split satisfying BOTH lever rules for a
specified average composition (PHI_BAR, ELL_BAR).

This is a mean-field associating-mixture model. It replaces the earlier
edge-linker convention in which bound linkers did not count toward the
total concentration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, least_squares


# ============================================================
# USER INPUTS
# ============================================================

PHI_BAR = 3.0e-5
ELL_BAR = 1.0e-5

# The plotted binding-strength interval.
BETA_AXIS_MIN = 0.0
BETA_AXIS_MAX = 20.0
N_BETA = 51

# False: plotted energy is for one complete two-ended bridge.
# True: plotted energy is per sticky end, assuming two identical,
# additive sticky ends, so beta*epsilon_bridge = 2 beta*epsilon_end.
PLOT_ENERGY_PER_END = False

# Standard-state factor in the mass-action equation:
#
# m = K0 exp(beta epsilon_bridge) (ell-m) (2 phi-m)^2
#
# K0=1 corresponds to the dimensionless lattice standard state.
K0 = 1.0

# Global optimization controls. Increase these for a more exhaustive search.
DE_MAXITER = 260
DE_POPSIZE = 14
DE_RESTARTS = 2
RANDOM_SEED = 12345

# A split is accepted only if it lowers the free energy by more than this.
FREE_ENERGY_TOL = 1.0e-11

OUTPUT_PREFIX = "all_linkers_total_concentration"


# ============================================================
# BASIC FUNCTIONS
# ============================================================

TINY = 1.0e-300
COMPOSITION_EPS = 1.0e-12


def xlogx(x: float) -> float:
    """Return x log x with the continuous value 0 at x=0."""
    if x <= 0.0:
        return 0.0
    return x * math.log(x)


def beta_bridge_from_axis(beta_axis: float) -> float:
    """Convert plotted energy to complete-bridge energy."""
    return 2.0 * beta_axis if PLOT_ENERGY_PER_END else beta_axis


def bridge_equilibrium(phi: float, ell: float, beta_bridge: float) -> float:
    """
    Solve the unique mass-action equation

        m = K0 exp(beta_bridge) (ell-m) (2 phi-m)^2

    on 0 <= m <= min(ell, 2 phi).
    """
    if phi <= 0.0 or ell <= 0.0:
        return 0.0

    upper = min(ell, 2.0 * phi)
    if upper <= 0.0:
        return 0.0

    log_activity = math.log(K0) + beta_bridge

    def log_residual(m: float) -> float:
        return (
            math.log(m)
            - math.log(ell - m)
            - 2.0 * math.log(2.0 * phi - m)
            - log_activity
        )

    # The residual tends to -infinity at m=0 and +infinity at the
    # physically active upper boundary.
    lower = max(TINY, upper * 1.0e-14)
    upper_open = upper * (1.0 - 1.0e-13)

    if log_residual(lower) > 0.0:
        lower = TINY

    if log_residual(upper_open) < 0.0:
        # Root is numerically indistinguishable from saturation.
        return upper_open

    return brentq(
        log_residual,
        lower,
        upper_open,
        xtol=1.0e-14,
        rtol=1.0e-12,
        maxiter=250,
    )


def free_energy(phi: float, ell: float, beta_bridge: float) -> float:
    """
    Minimized dimensionless homogeneous free energy beta*f.

    Before minimizing over m:

      beta f =
          phi ln phi + ell ln ell + s ln s
        + [m ln m + (ell-m) ln(ell-m) - ell ln ell]
        + 2[(2phi-m)ln(2phi-m) - 2phi ln(2phi)]
        + 2m
        - beta*epsilon_bridge*m

    The bracketed linker term partitions ALL linkers into free and bound
    populations. At m=0, the association contribution vanishes and the
    ordinary ideal mixing free energy is recovered.
    """
    solvent = 1.0 - phi - ell
    if (
        phi <= 0.0
        or ell <= 0.0
        or solvent <= 0.0
        or phi >= 1.0
        or ell >= 1.0
    ):
        return math.inf

    m = bridge_equilibrium(phi, ell, beta_bridge)

    mixing = xlogx(phi) + xlogx(ell) + xlogx(solvent)

    association = (
        xlogx(m)
        + xlogx(ell - m)
        - xlogx(ell)
        + 2.0 * (xlogx(2.0 * phi - m) - xlogx(2.0 * phi))
        + 2.0 * m
        - beta_bridge * m
        - m * math.log(K0)
    )

    return mixing + association


def thermodynamics(
    phi: float,
    ell: float,
    beta_bridge: float,
) -> tuple[float, float, float, float]:
    """
    Return (mu_phi, mu_ell, tangent-plane intercept, m).

    The envelope theorem allows derivatives at the equilibrium m:

      mu_phi = ln(phi/s) + 4 ln[(2phi-m)/(2phi)]
      mu_ell = ln[(ell-m)/s]
    """
    solvent = 1.0 - phi - ell
    if phi <= 0.0 or ell <= 0.0 or solvent <= 0.0:
        raise ValueError("Composition is outside the simplex.")

    m = bridge_equilibrium(phi, ell, beta_bridge)
    mu_phi = (
        math.log(phi / solvent)
        + 4.0 * math.log((2.0 * phi - m) / (2.0 * phi))
    )
    mu_ell = math.log((ell - m) / solvent)

    f = free_energy(phi, ell, beta_bridge)
    intercept = f - mu_phi * phi - mu_ell * ell

    return mu_phi, mu_ell, intercept, m


# ============================================================
# TWO-PHASE PARAMETERIZATION
# ============================================================

@dataclass
class SplitState:
    alpha_phase_2: float
    phi_1: float
    ell_1: float
    phi_2: float
    ell_2: float


def decode_split(
    variables: np.ndarray,
    phi_bar: float,
    ell_bar: float,
) -> SplitState:
    """
    Enforce both lever rules exactly.

    Variables are:
      alpha : fraction of phase 2
      u     : fraction of total A assigned to phase 1
      v     : fraction of total linker assigned to phase 1

    Then:
      (1-alpha) phi_1 + alpha phi_2 = phi_bar
      (1-alpha) ell_1 + alpha ell_2 = ell_bar
    """
    alpha, u, v = map(float, variables)

    phi_1 = phi_bar * u / (1.0 - alpha)
    phi_2 = phi_bar * (1.0 - u) / alpha

    ell_1 = ell_bar * v / (1.0 - alpha)
    ell_2 = ell_bar * (1.0 - v) / alpha

    return SplitState(alpha, phi_1, ell_1, phi_2, ell_2)


def valid_phase(phi: float, ell: float) -> bool:
    return (
        phi > COMPOSITION_EPS
        and ell > COMPOSITION_EPS
        and phi + ell < 1.0 - COMPOSITION_EPS
    )


def split_objective(
    variables: np.ndarray,
    phi_bar: float,
    ell_bar: float,
    beta_bridge: float,
) -> float:
    """Total free energy of a two-phase state."""
    state = decode_split(variables, phi_bar, ell_bar)

    if not valid_phase(state.phi_1, state.ell_1):
        excess = max(
            0.0,
            state.phi_1 + state.ell_1 - 1.0,
            -state.phi_1,
            -state.ell_1,
        )
        return 1.0e3 + 1.0e5 * excess

    if not valid_phase(state.phi_2, state.ell_2):
        excess = max(
            0.0,
            state.phi_2 + state.ell_2 - 1.0,
            -state.phi_2,
            -state.ell_2,
        )
        return 1.0e3 + 1.0e5 * excess

    return (
        (1.0 - state.alpha_phase_2)
        * free_energy(state.phi_1, state.ell_1, beta_bridge)
        + state.alpha_phase_2
        * free_energy(state.phi_2, state.ell_2, beta_bridge)
    )


def tangent_equation_residuals(
    variables: np.ndarray,
    phi_bar: float,
    ell_bar: float,
    beta_bridge: float,
) -> np.ndarray:
    """
    Three common-tangent-plane equations.

    The two lever rules are already enforced exactly by decode_split(), so
    only three residuals remain:
      mu_phi^(1) = mu_phi^(2)
      mu_ell^(1) = mu_ell^(2)
      omega^(1) = omega^(2)
    """
    state = decode_split(variables, phi_bar, ell_bar)

    if (
        not valid_phase(state.phi_1, state.ell_1)
        or not valid_phase(state.phi_2, state.ell_2)
    ):
        return np.array([1.0e3, 1.0e3, 1.0e3])

    mu_phi_1, mu_ell_1, omega_1, _ = thermodynamics(
        state.phi_1,
        state.ell_1,
        beta_bridge,
    )
    mu_phi_2, mu_ell_2, omega_2, _ = thermodynamics(
        state.phi_2,
        state.ell_2,
        beta_bridge,
    )

    # Scale the plane-intercept residual so it is not numerically
    # underweighted for very dilute average compositions.
    plane_scale = max(phi_bar + ell_bar, 1.0e-3)

    return np.array([
        mu_phi_1 - mu_phi_2,
        mu_ell_1 - mu_ell_2,
        (omega_1 - omega_2) / plane_scale,
    ])


def refine_common_tangent(
    variables: np.ndarray,
    phi_bar: float,
    ell_bar: float,
    beta_bridge: float,
) -> np.ndarray:
    """Refine a nontrivial split by solving the tangent-plane equations."""
    state = decode_split(variables, phi_bar, ell_bar)
    gap = (
        abs(state.phi_2 - state.phi_1)
        + abs(state.ell_2 - state.ell_1)
    )

    if gap < 1.0e-7:
        return variables

    result = least_squares(
        tangent_equation_residuals,
        variables,
        args=(phi_bar, ell_bar, beta_bridge),
        bounds=(
            np.array([1.0e-8, 1.0e-10, 1.0e-10]),
            np.array([1.0 - 1.0e-8, 1.0 - 1.0e-10, 1.0 - 1.0e-10]),
        ),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=5000,
    )

    refined = np.asarray(result.x, dtype=float)
    refined_state = decode_split(refined, phi_bar, ell_bar)

    if (
        valid_phase(refined_state.phi_1, refined_state.ell_1)
        and valid_phase(refined_state.phi_2, refined_state.ell_2)
    ):
        return refined

    return variables


def solve_split(
    phi_bar: float,
    ell_bar: float,
    beta_bridge: float,
    previous_solution: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, float, SplitState]:
    """Global multistart search for the best two-phase decomposition."""
    bounds = [
        (1.0e-8, 1.0 - 1.0e-8),  # alpha
        (1.0e-10, 1.0 - 1.0e-10),  # u
        (1.0e-10, 1.0 - 1.0e-10),  # v
    ]

    candidates: list[tuple[float, np.ndarray]] = []

    for restart in range(DE_RESTARTS):
        kwargs = dict(
            func=split_objective,
            bounds=bounds,
            args=(phi_bar, ell_bar, beta_bridge),
            seed=seed + restart,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=1.0e-9,
            polish=True,
            updating="immediate",
            workers=1,
        )

        if previous_solution is not None and restart == 0:
            kwargs["x0"] = previous_solution

        result = differential_evolution(**kwargs)
        candidates.append((float(result.fun), np.asarray(result.x, dtype=float)))

    best_energy, best_variables = min(candidates, key=lambda item: item[0])

    refined_variables = refine_common_tangent(
        best_variables,
        phi_bar,
        ell_bar,
        beta_bridge,
    )
    refined_energy = split_objective(
        refined_variables,
        phi_bar,
        ell_bar,
        beta_bridge,
    )

    # A tangent-plane root may represent a metastable split. Keep it only
    # when it is not higher in free energy than the global-search result.
    if refined_energy <= best_energy + 1.0e-8:
        best_variables = refined_variables
        best_energy = refined_energy

    best_state = decode_split(best_variables, phi_bar, ell_bar)

    return best_variables, best_energy, best_state


def ordered_phases(
    state: SplitState,
) -> tuple[float, float, float, float, float]:
    """
    Return dilute and dense endpoints ordered by A concentration:

      phi_low, ell_low, phi_high, ell_high, dense_fraction
    """
    if state.phi_1 <= state.phi_2:
        return (
            state.phi_1,
            state.ell_1,
            state.phi_2,
            state.ell_2,
            state.alpha_phase_2,
        )

    return (
        state.phi_2,
        state.ell_2,
        state.phi_1,
        state.ell_1,
        1.0 - state.alpha_phase_2,
    )


# ============================================================
# MAIN SWEEP
# ============================================================

def main() -> None:
    if not valid_phase(PHI_BAR, ELL_BAR):
        raise ValueError(
            "The average composition must satisfy "
            "PHI_BAR > 0, ELL_BAR > 0, and PHI_BAR + ELL_BAR < 1."
        )

    beta_axis_values = np.linspace(
        BETA_AXIS_MIN,
        BETA_AXIS_MAX,
        N_BETA,
    )

    # Sweep downward from strong binding so the previous nontrivial split
    # can seed the next calculation.
    solve_order = beta_axis_values[::-1]

    rows: list[dict[str, float | bool]] = []
    previous_solution: np.ndarray | None = None

    for index, beta_axis in enumerate(solve_order):
        beta_bridge = beta_bridge_from_axis(float(beta_axis))

        homogeneous_energy = free_energy(PHI_BAR, ELL_BAR, beta_bridge)

        variables, two_phase_energy, state = solve_split(
            PHI_BAR,
            ELL_BAR,
            beta_bridge,
            previous_solution,
            RANDOM_SEED + 100 * index,
        )
        previous_solution = variables

        phi_low, ell_low, phi_high, ell_high, dense_fraction = (
            ordered_phases(state)
        )

        composition_gap = (
            abs(phi_high - phi_low)
            + abs(ell_high - ell_low)
        )
        delta_f = two_phase_energy - homogeneous_energy

        separated = (
            delta_f < -FREE_ENERGY_TOL
            and composition_gap > 1.0e-7
        )

        if separated:
            mu_phi_low, mu_ell_low, intercept_low, m_low = thermodynamics(
                phi_low, ell_low, beta_bridge
            )
            mu_phi_high, mu_ell_high, intercept_high, m_high = thermodynamics(
                phi_high, ell_high, beta_bridge
            )

            phi_lever_residual = (
                (1.0 - dense_fraction) * phi_low
                + dense_fraction * phi_high
                - PHI_BAR
            )
            ell_lever_residual = (
                (1.0 - dense_fraction) * ell_low
                + dense_fraction * ell_high
                - ELL_BAR
            )

            mu_phi_residual = mu_phi_low - mu_phi_high
            mu_ell_residual = mu_ell_low - mu_ell_high
            plane_residual = intercept_low - intercept_high
        else:
            phi_low = math.nan
            ell_low = math.nan
            phi_high = math.nan
            ell_high = math.nan
            dense_fraction = 0.0
            m_low = math.nan
            m_high = math.nan
            phi_lever_residual = math.nan
            ell_lever_residual = math.nan
            mu_phi_residual = math.nan
            mu_ell_residual = math.nan
            plane_residual = math.nan

        rows.append({
            "beta_axis": float(beta_axis),
            "beta_epsilon_bridge": beta_bridge,
            "phase_separated": separated,
            "delta_f_two_minus_homogeneous": delta_f,
            "phi_low": phi_low,
            "ell_low_total": ell_low,
            "m_bound_low": m_low,
            "ell_free_low": ell_low - m_low if separated else math.nan,
            "solvent_low": 1.0 - phi_low - ell_low if separated else math.nan,
            "phi_high": phi_high,
            "ell_high_total": ell_high,
            "m_bound_high": m_high,
            "ell_free_high": ell_high - m_high if separated else math.nan,
            "solvent_high": 1.0 - phi_high - ell_high if separated else math.nan,
            "dense_phase_fraction": dense_fraction,
            "phi_lever_residual": phi_lever_residual,
            "ell_lever_residual": ell_lever_residual,
            "mu_phi_residual": mu_phi_residual,
            "mu_ell_residual": mu_ell_residual,
            "tangent_plane_residual": plane_residual,
        })

        print(
            f"beta axis={beta_axis:8.4f} | "
            f"separated={str(separated):5s} | "
            f"delta f={delta_f:+.4e}"
        )

    data = (
        pd.DataFrame(rows)
        .sort_values("beta_axis")
        .reset_index(drop=True)
    )

    csv_path = Path(f"{OUTPUT_PREFIX}_data.csv")
    data.to_csv(csv_path, index=False)

    separated_data = data[data["phase_separated"]].copy()

    if separated_data.empty:
        print("\nNo two-phase state was found in the requested range.")
        print("Increase BETA_AXIS_MAX or the optimization settings.")
        print(f"Data saved to {csv_path}")
        return

    entry = separated_data.iloc[0]
    energy_name = (
        "beta*epsilon_end"
        if PLOT_ENERGY_PER_END
        else "beta*epsilon_bridge"
    )

    print("\nFirst detected compatible two-phase state:")
    print(f"  {energy_name:24s} = {entry['beta_axis']:.10g}")
    print(f"  phi_-                    = {entry['phi_low']:.10g}")
    print(f"  ell_- total              = {entry['ell_low_total']:.10g}")
    print(f"  phi_+                    = {entry['phi_high']:.10g}")
    print(f"  ell_+ total              = {entry['ell_high_total']:.10g}")
    print(
        f"  dense phase fraction     = "
        f"{entry['dense_phase_fraction']:.10g}"
    )

    print("\nMaximum absolute residuals in separated states:")
    for column in [
        "phi_lever_residual",
        "ell_lever_residual",
        "mu_phi_residual",
        "mu_ell_residual",
        "tangent_plane_residual",
    ]:
        print(f"  {column:24s}: {separated_data[column].abs().max():.3e}")

    ylabel = (
        r"Per-end binding strength, $\beta\epsilon_{\rm end}$"
        if PLOT_ENERGY_PER_END
        else r"Bridge binding strength, $\beta\epsilon_{\rm bridge}$"
    )

    # Plot 1: full compatible binodal in phi-beta plane.
    plt.xscale('log')
    plt.figure(figsize=(7.5, 5.7))
    plt.plot(
        separated_data["phi_low"],
        separated_data["beta_axis"],
        label=r"Dilute endpoint $\phi_-$",
    )
    plt.plot(
        separated_data["phi_high"],
        separated_data["beta_axis"],
        label=r"Dense endpoint $\phi_+$",
    )
    plt.axvline(
        PHI_BAR,
        linestyle="--",
        label=rf"Average $\bar{{\phi}}={PHI_BAR:g}$",
    )
    plt.xlabel(r"Total $A$ fraction, $\phi$")
    plt.ylabel(ylabel)
    plt.title(
        rf"Compatible binodal, "
        rf"$(\bar{{\phi}},\bar{{\ell}})=({PHI_BAR:g},{ELL_BAR:g})$"
    )
    plt.xlim(0.0, max(0.2, 1.05 * separated_data["phi_high"].max()))
    plt.legend()
    plt.tight_layout()
    phi_plot_path = Path(f"{OUTPUT_PREFIX}_phi_binodal.png")
    plt.savefig(phi_plot_path, dpi=220, bbox_inches="tight")
    plt.show()

    # Plot 2: total linker endpoint fractions. These must remain below 1.
    plt.figure(figsize=(7.5, 5.7))
    plt.plot(
        separated_data["ell_low_total"],
        separated_data["beta_axis"],
        label=r"Dilute endpoint $\ell_-$",
    )
    plt.plot(
        separated_data["ell_high_total"],
        separated_data["beta_axis"],
        label=r"Dense endpoint $\ell_+$",
    )
    plt.axvline(
        ELL_BAR,
        linestyle="--",
        label=rf"Average $\bar{{\ell}}={ELL_BAR:g}$",
    )
    plt.xlabel(r"Total linker fraction, $\ell$")
    plt.ylabel(ylabel)
    plt.title("All linkers included in total concentration")
    plt.xlim(0.0, min(1.0, 1.05 * separated_data["ell_high_total"].max()))
    plt.legend()
    plt.tight_layout()
    ell_plot_path = Path(f"{OUTPUT_PREFIX}_ell_binodal.png")
    plt.savefig(ell_plot_path, dpi=220, bbox_inches="tight")
    plt.show()

    # Plot 3: dense-phase fraction.
    plt.figure(figsize=(7.5, 5.7))
    plt.plot(
        separated_data["beta_axis"],
        separated_data["dense_phase_fraction"],
    )
    plt.xlabel(ylabel)
    plt.ylabel(r"Dense-phase volume fraction, $\alpha$")
    plt.title("Lever-rule dense-phase fraction")
    plt.tight_layout()
    alpha_plot_path = Path(f"{OUTPUT_PREFIX}_dense_fraction.png")
    plt.savefig(alpha_plot_path, dpi=220, bbox_inches="tight")
    plt.show()

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {phi_plot_path}")
    print(f"  {ell_plot_path}")
    print(f"  {alpha_plot_path}")


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parent-constrained binodal for linker lattice model
#
# Inputs are experimental parent concentrations:
#   c_ns_parent_uM
#   c_l_parent_uM
#
# Internally:
#   phi0 = V_n c_ns / c0
#   rho_l0 = V_l c_l / c0
#   R0 = 2 rho_l0 / z
#
# For each binodal tie line:
#   phi_d = p
#   phi_D = 1 - p
#
# We enforce both:
#   phi0 = (1-alpha) phi_d + alpha phi_D
#   R0   = (1-alpha) R(phi_d) + alpha R(phi_D)
#
# This determines the linker activity y required for that tie line.
# Then K is obtained from q = (1+yK)/(1+y).
# Finally T is obtained from K(T).
# ============================================================


# -----------------------------
# Experimental / model values
# -----------------------------
z = 4
c0_M = 55.5

# Effective coarse-grained lattice volumes.
# These are not literal geometric volumes.
# V_n = V_l = 1e7 makes:
#   c_ns = 0.5 uM -> phi0 ≈ 0.09
#   c_l  = 1.0 uM -> rho_l0 ≈ 0.18
V_n = 3.0e6
V_l = 1.0e6

# Experimental parent nanostar concentration
c_ns_parent_uM = 0.5

# Linker scan around the experimental 2:1 linker:NS baseline
parent_linker_concentration_values_uM = [0.5, 1.0, 1.5, 2]

# Effective thermodynamics
delta_H_kJ_per_mol = -80.0
delta_S_kJ_per_mol_K = -0.210
R_gas_kJ_per_mol_K = 0.008314462618

# Binodal parameter
p_grid = np.linspace(1e-4, 0.5 - 1e-6, 6000)


# -----------------------------
# Unit conversions
# -----------------------------
def phi_from_cns_uM(c_ns_uM):
    c_ns_M = c_ns_uM * 1.0e-6
    return V_n * c_ns_M / c0_M


def cns_uM_from_phi(phi):
    c_ns_M = c0_M * phi / V_n
    return 1.0e6 * c_ns_M


def rho_l_from_cl_uM(c_l_uM):
    """
    Dimensionless linker lattice density:
        rho_l = V_l c_l / c0
    """
    c_l_M = c_l_uM * 1.0e-6
    return V_l * c_l_M / c0_M


def cl_uM_from_rho_l(rho_l):
    c_l_M = rho_l * c0_M / V_l
    return 1.0e6 * c_l_M


def R_from_rho_l(rho_l):
    """
    Normalized linker density:
        R = 2 rho_l / z
    """
    return 2.0 * rho_l / z


def rho_l_from_R(R):
    return 0.5 * z * R


# -----------------------------
# Binodal formulas
# -----------------------------
def b_binodal(p):
    """
    b(p) = ln((1-p)/p) / (1 - 2p)
    """
    p = np.asarray(p, dtype=float)
    denom = 1.0 - 2.0 * p

    b = np.empty_like(p)
    near_half = np.isclose(denom, 0.0, atol=1e-8)

    b[near_half] = 2.0
    b[~near_half] = np.log((1.0 - p[~near_half]) / p[~near_half]) / denom[~near_half]

    return b


def q_from_p(p):
    """
    q(p) = exp((2/z) b(p))
    """
    return np.exp((2.0 / z) * b_binodal(p))


def K_from_q_and_y(q, y):
    """
    q = (1 + yK)/(1 + y)

    Therefore:
        K = [q(1+y)-1]/y
    """
    return (q * (1.0 + y) - 1.0) / y


def temperature_from_K(K):
    """
    K(T) = exp[-DeltaG(T)/(R T)]
    DeltaG(T) = DeltaH - T DeltaS

    Solving for T:
        T = DeltaH / (DeltaS - R ln K)
    """
    K = np.where((K > 0.0) & np.isfinite(K), K, np.nan)

    denominator = delta_S_kJ_per_mol_K - R_gas_kJ_per_mol_K * np.log(K)
    T_K = delta_H_kJ_per_mol / denominator

    return np.where((T_K > 0.0) & np.isfinite(T_K), T_K, np.nan)


def K_from_temperature_K(T_K):
    """
    Forward check:
        K(T) = exp[-(DeltaH - T DeltaS)/(R T)]
    """
    delta_G = delta_H_kJ_per_mol - T_K * delta_S_kJ_per_mol_K
    return np.exp(-delta_G / (R_gas_kJ_per_mol_K * T_K))


# -----------------------------
# Parent-constrained linker logic
# -----------------------------
def y_from_parent_phi_and_R(p, q, phi0, R0):
    """
    Solve for linker activity y along each tie line
    for a fixed parent nanostar volume fraction phi0
    and fixed parent normalized linker density R0.

    Tie line:
        phi_d = p
        phi_D = 1 - p

    Nanostar lever rule:
        phi0 = (1-alpha) phi_d + alpha phi_D

    Linker lever rule:
        R0 = (1-alpha) R(phi_d) + alpha R(phi_D)

    With:
        R(phi) = r_f + (r_b - r_f) phi^2

    and using q = (1+yK)/(1+y), one can eliminate K and solve for r_f.
    """
    phi_d = p
    phi_D = 1.0 - p

    alpha = (phi0 - phi_d) / (phi_D - phi_d)
    valid_alpha = (alpha >= 0.0) & (alpha <= 1.0)

    phi2_avg = (1.0 - alpha) * phi_d**2 + alpha * phi_D**2

    A = ((q - 1.0) / q) * phi2_avg

    r_f = (R0 - A) / (1.0 - A)

    y = r_f / (1.0 - r_f)

    valid_y = (r_f > 0.0) & (r_f < 1.0) & np.isfinite(y)

    return np.where(valid_alpha & valid_y, y, np.nan)


def parent_constrained_binodal(c_ns_parent_uM, c_l_parent_uM):
    """
    Main function.

    Given experimental parent concentrations:
        c_ns_parent_uM
        c_l_parent_uM

    Return the parent-constrained binodal:
        dilute concentration
        dense concentration
        temperature
        internal linker activity y
        K
        alpha
    """
    phi0 = phi_from_cns_uM(c_ns_parent_uM)
    rho_l0 = rho_l_from_cl_uM(c_l_parent_uM)
    R0 = R_from_rho_l(rho_l0)

    if not (0.0 < phi0 < 1.0):
        raise ValueError(f"Invalid phi0={phi0}. Need 0 < phi0 < 1.")

    if not (0.0 < R0 < 1.0):
        raise ValueError(f"Invalid R0={R0}. Need 0 < R0 < 1.")

    p = p_grid
    q = q_from_p(p)

    y = y_from_parent_phi_and_R(
        p=p,
        q=q,
        phi0=phi0,
        R0=R0,
    )

    K = K_from_q_and_y(q, y)
    T_K = temperature_from_K(K)
    T_C = T_K - 273.15

    phi_d = p
    phi_D = 1.0 - p

    alpha = (phi0 - phi_d) / (phi_D - phi_d)

    mask = (
        np.isfinite(y)
        & np.isfinite(K)
        & np.isfinite(T_C)
        & (K > 0.0)
        & (alpha >= 0.0)
        & (alpha <= 1.0)
    )

    return {
        "phi0": phi0,
        "rho_l0": rho_l0,
        "R0": R0,
        "c_ns_parent_uM": c_ns_parent_uM,
        "c_l_parent_uM": c_l_parent_uM,
        "phi_dilute": phi_d[mask],
        "phi_dense": phi_D[mask],
        "c_ns_dilute_uM": cns_uM_from_phi(phi_d[mask]),
        "c_ns_dense_uM": cns_uM_from_phi(phi_D[mask]),
        "temperature_C": T_C[mask],
        "temperature_K": T_K[mask],
        "K": K[mask],
        "y": y[mask],
        "alpha": alpha[mask],
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_parent_constrained_binodals():
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    for c_l_parent_uM in parent_linker_concentration_values_uM:
        out = parent_constrained_binodal(
            c_ns_parent_uM=c_ns_parent_uM,
            c_l_parent_uM=c_l_parent_uM,
        )

        cns_full = np.concatenate([
            out["c_ns_dilute_uM"],
            out["c_ns_dense_uM"][::-1],
        ])

        T_full = np.concatenate([
            out["temperature_C"],
            out["temperature_C"][::-1],
        ])

        ax.plot(
            cns_full,
            T_full,
            label=(
                rf"$\bar c_\ell={c_l_parent_uM:g}\ \mu$M, "
                rf"$\rho_{{\ell,0}}={out['rho_l0']:.3g}$"
            ),
        )

    ax.axvline(
        c_ns_parent_uM,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=rf"$\bar c_{{\rm ns}}={c_ns_parent_uM:g}\ \mu$M",
    )

    ax.axhspan(
        25.0,
        55.0,
        color="gray",
        alpha=0.12,
        label=r"experimental anneal window $25{-}55^\circ$C",
    )

    ax.set_xlabel(r"Coexisting nanostar concentration $c_{\rm ns}$ ($\mu$M)")
    ax.set_ylabel(r"Temperature $T$ ($^\circ$C)")
    ax.set_title("Parent-constrained binodal from experimental parent concentrations")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.show()


if __name__ == "__main__":
    print("Experimental baseline conversions:")
    phi0 = phi_from_cns_uM(c_ns_parent_uM)
    print(f"c_ns_parent = {c_ns_parent_uM:g} uM")
    print(f"phi0 = {phi0:.4g}")

    for c_l in parent_linker_concentration_values_uM:
        rho0 = rho_l_from_cl_uM(c_l)
        R0 = R_from_rho_l(rho0)
        print(f"c_l_parent = {c_l:g} uM -> rho_l0 = {rho0:.4g}, R0 = {R0:.4g}")

    plot_parent_constrained_binodals()
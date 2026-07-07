import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parent-projected binodal phase diagram for linker lattice model
#
# x-axis: parent nanostar composition phi_A
# y-axis: binodal temperature T
#
# Uses:
#   K = exp[-DeltaG(T)/(RT)]
#   DeltaG(T) = DeltaH - T DeltaS
#
# Therefore:
#   ln K = -DeltaH/(RT) + DeltaS/R
#
# Solving for T:
#   T = -DeltaH / [R (ln K - DeltaS/R)]
# ============================================================


# -----------------------------
# Model parameters
# -----------------------------
z = 4  # nanostar valence

# Parent linker densities.
# For z = 4, v_l = z/2 = 2, so rho_bar should usually be between 0 and 2.
rho_bar_values = [0.5, 1.0, 1.5, 2.0]


# -----------------------------
# Thermodynamic parameters
# -----------------------------
# DeltaG(T) = DeltaH - T DeltaS
#
# Units:
#   DeltaH: kJ/mol
#   DeltaS: kJ/(mol K)
#   R:      kJ/(mol K)
#
# These are illustrative values.
# Replace with your experimental or fitted values.
delta_H_kJ_per_mol = -800.0
delta_S_kJ_per_mol_K = -0.220

R_gas_kJ = 0.008314462618  # kJ/(mol K)


# -----------------------------
# Composition grid
# -----------------------------
phi = np.linspace(1e-4, 1 - 1e-4, 4000)


def q_of_p(p, z):
    """
    Computes

        q(p) = exp[(2/z) b(p)]

    where

        b(p) = ln((1-p)/p) / (1 - 2p)

    and p = min(phi, 1-phi).

    At p = 1/2, the limiting value is b = 2.
    """
    p = np.asarray(p)
    b = np.empty_like(p)

    near_half = np.isclose(p, 0.5, atol=1e-6)

    b[near_half] = 2.0

    b[~near_half] = (
        np.log((1 - p[~near_half]) / p[~near_half])
        / (1 - 2 * p[~near_half])
    )

    q = np.exp((2 / z) * b)

    return q


def K_binodal(phi, rho_bar, z):
    """
    Computes the parent-projected binodal K value:

        K_binodal(phi; R)
        =
        q(p) * [R + (q(p)-1)(1-phi^2)]
        /
        [q(p)R - (q(p)-1)phi^2]

    where

        R = 2 rho_bar / z

    and

        p = min(phi, 1-phi).
    """
    R = 2 * rho_bar / z

    p = np.minimum(phi, 1 - phi)
    q = q_of_p(p, z)

    numerator = q * (R + (q - 1) * (1 - phi**2))
    denominator = q * R - (q - 1) * phi**2

    K = numerator / denominator

    # Keep only physically meaningful attractive-binding values.
    # K > 1 corresponds to favorable binding in this convention.
    K = np.where((denominator > 0) & (K > 1), K, np.nan)

    return K


def temperature_from_K_vant_hoff(K, delta_H, delta_S):
    """
    Convert K_binodal into temperature using

        K = exp[-DeltaG(T)/(RT)]

    with

        DeltaG(T) = DeltaH - T DeltaS.

    Then

        ln K = -DeltaH/(RT) + DeltaS/R

    so

        T = -DeltaH / [R (ln K - DeltaS/R)].
    """
    denominator = R_gas_kJ * (np.log(K) - delta_S / R_gas_kJ)

    T = -delta_H / denominator

    return T


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

for rho_bar in rho_bar_values:
    K = K_binodal(phi, rho_bar, z)

    T = temperature_from_K_vant_hoff(
        K,
        delta_H_kJ_per_mol,
        delta_S_kJ_per_mol_K
    )

    # Optional plotting filter.
    # Adjust this depending on the physical temperature range you care about.
    T = np.where((T > 200) & (T < 500), T, np.nan)

    plt.plot(
        phi,
        T,
        label=rf"$\bar\rho_\ell = {rho_bar}$"
    )


plt.xlabel(r"Parent composition $\bar\phi_A$")
plt.ylabel(r"Binodal temperature $T$ (K)")
plt.title(r"Parent-projected binodal with $\Delta G(T)=\Delta H-T\Delta S$")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
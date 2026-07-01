import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# Parameters
# ============================================================

z = 4
beta_deltaG = -13

phi_min = 1e-4
phi_max = 0.999
n_phi = 300

nu0 = 0.095
nu_min = 1e-4
nu_max = 10

# ============================================================
# Original no-volume model
# ============================================================

def epsilon(phi):
    """
    Optional direct nanostar energetic term per lattice site.
    Keep zero for pure linker-mediated model.
    """
    return 0.0

def phi0_from_phi(phi):
    return 1.0 - phi

def m_from_phi(phi):
    """
    A--A contact-slot density:
    m_AA = z phi_A^2 / 2
    """
    return 0.5 * z * phi**2

def ell_star(nu, m):
    """
    Exact saddle bridge density ell_*.
    """
    q = np.exp(beta_deltaG)
    d = np.sqrt((nu - m)**2 + 4.0 * q * nu * m)
    return 2.0 * nu * m / (nu + m + d)

def beta_f_raw(phi, nu):
    """
    Raw dimensionless free-energy density beta f per lattice site.
    """
    eps = 1e-14

    phi0 = phi0_from_phi(phi)
    m = m_from_phi(phi)
    ell = ell_star(nu, m)

    valid = (
        (phi > 0)
        & (phi0 > 0)
        & (nu > 0)
        & (m > 0)
        & ((nu - ell) > 0)
        & ((m - ell) > 0)
    )

    f = np.full_like(phi, np.nan, dtype=float)
    print(nu)
    f[valid] = (
        # epsilon(phi[valid])
        + phi[valid] * np.log(np.maximum(phi[valid], eps))
        + phi0[valid] * np.log(np.maximum(phi0[valid], eps))

        # Ideal linker entropy.
        # For fixed nu, this is just a vertical shift.
        # + nu * np.log(np.maximum(nu, eps))

        # Linker/contact association terms.
        + nu * np.log(np.maximum((nu - ell[valid]), eps))
        + m[valid] * np.log(np.maximum((m[valid] - ell[valid]) / m[valid], eps))
    )

    return f

def beta_f_mix(phi, nu):
    """
    Referenced mixing free energy:

        Delta f(phi) = f(phi) - (1-phi) f(0) - phi f(1)

    This is the correct analogue of
        g_AB - (g_AA + g_BB)/2.
    """
    f = beta_f_raw(phi, nu)

    # Use tiny offsets because logs at exactly 0 or 1 are singular.
    phi_left = np.array([phi_min])
    phi_right = np.array([phi_max])

    f_left = beta_f_raw(phi_left, nu)[0]
    f_right = beta_f_raw(phi_right, nu)[0]

    f_ref = (1.0 - phi) * f_left + phi * f_right

    return f - f_ref

def second_derivative(y, x):
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    return dy

def find_spinodal_points(phi, f2):
    roots = []

    for k in range(len(phi) - 1):
        y1 = f2[k]
        y2 = f2[k + 1]

        if not np.isfinite(y1) or not np.isfinite(y2):
            continue

        if y1 == 0:
            roots.append(phi[k])

        if y1 * y2 < 0:
            x1 = phi[k]
            x2 = phi[k + 1]
            root = x1 - y1 * (x2 - x1) / (y2 - y1)
            roots.append(root)

    return roots

# ============================================================
# Grid
# ============================================================

phi = np.linspace(phi_min, phi_max, n_phi)

# ============================================================
# Plot
# ============================================================

fig, (ax_f, ax_f2) = plt.subplots(
    2,
    1,
    figsize=(9, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

plt.subplots_adjust(bottom=0.18)

def plot_for_nu(nu):
    ax_f.clear()
    ax_f2.clear()

    f = beta_f_mix(phi, nu)
    f2 = second_derivative(f, phi)

    spinodal_roots = find_spinodal_points(phi, f2)
    unstable = f2 < 0

    ax_f.plot(phi, f, linewidth=2)

    # ax_f.set_xscale('log')

    ax_f.fill_between(
        phi,
        np.nanmin(f),
        np.nanmax(f),
        where=unstable,
        alpha=0.18,
        label=r"$d^2\Delta(\beta f)/d\phi_A^2<0$"
    )

    for root in spinodal_roots:
        ax_f.axvline(root, linestyle="--", linewidth=1.5)

    ax_f.axhline(0.0, linestyle=":", linewidth=1.2)

    ax_f.set_ylabel(r"$\Delta(\beta f)$")
    ax_f.set_title(
        rf"Referenced mixing free energy, fixed $\nu_L={nu:.3f}$, "
        rf"$\beta\Delta G={beta_deltaG:.2f}$"
    )
    ax_f.legend(loc="best")
    ax_f.grid(True)

    ax_f2.plot(phi, f2, linewidth=2)
    ax_f2.axhline(0.0, linestyle="--", linewidth=1.2)

    for root in spinodal_roots:
        ax_f2.axvline(root, linestyle="--", linewidth=1.5)

    ax_f2.set_xlabel(r"$\phi_A$")
    ax_f2.set_ylabel(r"$d^2\Delta(\beta f)/d\phi_A^2$")
    ax_f2.grid(True)

    if spinodal_roots:
        print("Spinodal boundary points:")
        # for root in spinodal_roots:
        #     print(f"phi_A = {root:.6f}")
    else:
        print("No spinodal points for this nu_L.")

plot_for_nu(nu0)

# ============================================================
# Slider for nu_L
# ============================================================

slider_ax = plt.axes([0.20, 0.06, 0.60, 0.03])

nu_slider = Slider(
    slider_ax,
    r"$\nu_L$",
    nu_min,
    nu_max,
    valinit=nu0
)

def update(val):
    nu = nu_slider.val
    plot_for_nu(nu)
    fig.canvas.draw_idle()

nu_slider.on_changed(update)

plt.show()
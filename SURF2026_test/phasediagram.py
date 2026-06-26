import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull


# -----------------------------
# User parameters
# -----------------------------
chi12 = -2
chi1s = 3.0
chi2s = 3.0

N1 = 50          # polymerization degree / size of A
N2 = 50          # polymerization degree / size of B
Ns = 1           # solvent size

grid_n = 160
eps = 1e-8
dominance_ratio = 3.0


# -----------------------------
# Flory-Huggins free energy
# -----------------------------
def fh_free_energy(phi1, phi2):
    """
    Ternary FH free energy with solvent eliminated:
        phi_s = 1 - phi1 - phi2
    """
    phis = 1.0 - phi1 - phi2

    return (
        (phi1 / N1) * np.log(phi1)
        + (phi2 / N2) * np.log(phi2)
        + (phis / Ns) * np.log(phis)
        + chi12 * phi1 * phi2
        + chi1s * phi1 * phis
        + chi2s * phi2 * phis
    )


def fh_hessian(phi1, phi2):
    phis = 1.0 - phi1 - phi2

    H11 = 1.0 / (N1 * phi1) + 1.0 / (Ns * phis) - 2.0 * chi1s
    H22 = 1.0 / (N2 * phi2) + 1.0 / (Ns * phis) - 2.0 * chi2s
    H12 = 1.0 / (Ns * phis) + chi12 - chi1s - chi2s

    return np.array([[H11, H12], [H12, H22]])


def smallest_hessian_eigenpair(phi1, phi2):
    H = fh_hessian(phi1, phi2)
    vals, vecs = np.linalg.eigh(H)
    return vals[0], vecs[:, 0]


# -----------------------------
# Ternary coordinates
# -----------------------------
def ternary_to_xy(phi1, phi2):
    """
    Triangle vertices:
        A       = (0, 0)
        B       = (1, 0)
        solvent = (1/2, sqrt(3)/2)
    """
    phis = 1.0 - phi1 - phi2
    x = phi2 + 0.5 * phis
    y = (np.sqrt(3.0) / 2.0) * phis
    return x, y


def make_simplex_grid(grid_n):
    phi1_list = []
    phi2_list = []

    for i in range(1, grid_n):
        for j in range(1, grid_n - i):
            phi1 = i / grid_n
            phi2 = j / grid_n
            phis = 1.0 - phi1 - phi2

            if phi1 > eps and phi2 > eps and phis > eps:
                phi1_list.append(phi1)
                phi2_list.append(phi2)

    return np.array(phi1_list), np.array(phi2_list)


# -----------------------------
# Binodal approximation
# -----------------------------
def lower_convex_envelope(phi1, phi2, f):
    """
    Approximate the convex envelope of f(phi1, phi2).

    We build the 3D surface:
        (phi1, phi2, f)

    The lower convex hull gives the equilibrium free-energy envelope.
    If f is above this envelope, the homogeneous state is not globally stable.
    """
    points_3d = np.column_stack([phi1, phi2, f])
    hull = ConvexHull(points_3d, qhull_options="QJ")

    lower_planes = []

    for eq in hull.equations:
        a, b, c, d = eq

        # Lower hull facets have downward-facing normal.
        if c < -1e-12:
            lower_planes.append(eq)

    envelope = np.full_like(f, -np.inf, dtype=float)

    for a, b, c, d in lower_planes:
        plane_z = -(a * phi1 + b * phi2 + d) / c
        envelope = np.maximum(envelope, plane_z)

    return envelope


# -----------------------------
# Spinodal blob classification
# -----------------------------
def classify_spinodal_blob(phi1, phi2):
    lam, v = smallest_hessian_eigenpair(phi1, phi2)

    if lam >= 0:
        return 0  # no blobs / locally stable

    v1, v2 = v
    a, b = abs(v1), abs(v2)

    # Same sign means A and B enrich together against solvent.
    if v1 * v2 > 0:
        if a > dominance_ratio * b:
            return 1  # A-rich blob
        if b > dominance_ratio * a:
            return 2  # B-rich blob
        return 3      # A+B blob

    # Opposite signs means A and B demix from each other.
    return 4          # separate A and B blobs


# -----------------------------
# Plot helpers
# -----------------------------
def decorate_ternary_axes(ax):
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3.0) / 2.0, 0]

    ax.plot(triangle_x, triangle_y, linewidth=1.5)

    ax.text(-0.04, -0.04, r"$\phi_1=1$  A", ha="right", va="top")
    ax.text(1.04, -0.04, r"$\phi_2=1$  B", ha="left", va="top")
    ax.text(
        0.5,
        np.sqrt(3.0) / 2.0 + 0.04,
        r"$\phi_s=1$  solvent",
        ha="center",
        va="bottom",
    )

    ax.set_aspect("equal")
    ax.axis("off")


def add_parameter_title(ax, prefix):
    ax.set_title(
        prefix
        + "\n"
        + rf"$\chi_{{12}}={chi12}$, "
        + rf"$\chi_{{1s}}={chi1s}$, "
        + rf"$\chi_{{2s}}={chi2s}$, "
        + rf"$N_1={N1}$, $N_2={N2}$, $N_s={Ns}$"
    )


# -----------------------------
# Main computation
# -----------------------------
phi1, phi2 = make_simplex_grid(grid_n)
phis = 1.0 - phi1 - phi2
f = fh_free_energy(phi1, phi2)

x, y = ternary_to_xy(phi1, phi2)
triang = Triangulation(x, y)

# Spinodal
min_eig = np.array([
    smallest_hessian_eigenpair(a, b)[0]
    for a, b in zip(phi1, phi2)
])

inside_spinodal = min_eig < 0

# Binodal / coexistence region
f_envelope = lower_convex_envelope(phi1, phi2, f)
gap = f - f_envelope

binodal_tol = 2e-4
inside_binodal = gap > binodal_tol

# Stability classification:
# 0 = stable
# 1 = metastable
# 2 = spinodal unstable
stability = np.zeros_like(f, dtype=int)
stability[inside_binodal] = 1
stability[inside_spinodal] = 2

# Blob classification
blob_class = np.array([
    classify_spinodal_blob(a, b)
    for a, b in zip(phi1, phi2)
])


# -----------------------------
# Figure 1: binodal + spinodal
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 7))

levels = [-0.5, 0.5, 1.5, 2.5]
cf = ax.tricontourf(triang, stability, levels=levels, alpha=0.85)

# Solid contour: approximate binodal boundary
ax.tricontour(triang, gap, levels=[binodal_tol], linewidths=1.8)

# Dashed contour: spinodal boundary
ax.tricontour(triang, min_eig, levels=[0.0], linewidths=1.8, linestyles="--")

decorate_ternary_axes(ax)
add_parameter_title(ax, "Ternary FH binodal and spinodal")

cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(["stable", "metastable", "spinodal unstable"])

fig.savefig("ternary_fh_binodal_spinodal.png", dpi=220, bbox_inches="tight")
plt.show()


# -----------------------------
# Figure 2: spinodal blob type
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 7))

levels = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
cf = ax.tricontourf(triang, blob_class, levels=levels, alpha=0.85)

ax.tricontour(triang, min_eig, levels=[0.0], linewidths=1.8, linestyles="--")

decorate_ternary_axes(ax)
add_parameter_title(ax, "Ternary FH spinodal blob type")

cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0, 1, 2, 3, 4])
cbar.set_ticklabels([
    "no blobs",
    "A-rich blob",
    "B-rich blob",
    "A+B blob",
    "separate A and B",
])

fig.savefig("ternary_fh_spinodal_blob_types.png", dpi=220, bbox_inches="tight")
plt.show()
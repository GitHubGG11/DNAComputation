import time
import numpy as np


def enforce_bounds(solutes, eps=1e-6):
    solutes = np.clip(solutes, eps, 1.0 - eps)

    total_solute = np.sum(solutes, axis=0)

    mask = total_solute > 1.0 - eps

    if np.any(mask):
        solutes[:, mask] *= (1.0 - eps) / total_solute[mask]

    solvent = 1.0 - np.sum(solutes, axis=0)
    solvent = np.clip(solvent, eps, 1.0)

    return solutes, solvent


def chemical_potential_bulk(solutes, chi, eps=1e-8):
    """
    solutes shape:
        n_solutes x cx x cy

    chi shape:
        N x N

    where:
        N = n_solutes + 1

    The last component is the solvent:

        solvent = 1 - sum(solutes)

    Returns:
        mu_bulk for the evolved solutes only,
        shape n_solutes x cx x cy.
    """

    solutes, solvent = enforce_bounds(solutes, eps=eps)

    n_solutes = solutes.shape[0]
    N = n_solutes + 1

    chi = np.asarray(chi)

    if chi.shape != (N, N):
        raise ValueError(
            f"chi must have shape {(N, N)}, but got {chi.shape}."
        )

    full_grid = np.concatenate(
        [
            solutes,
            solvent[None, :, :]
        ],
        axis=0
    )

    # If the interaction free energy is
    #
    #   f_int = 1/2 sum_ab chi_ab phi_a phi_b,
    #
    # then
    #
    #   d f_int / d phi_a = sum_b chi_ab phi_b
    #
    # when chi is symmetric.
    #
    # Symmetrizing makes this safer if only one side of chi is filled.

    chi_sym = 0.5 * (chi + chi.T)

    interaction_derivative = np.einsum(
        "ab,bxy->axy",
        chi_sym,
        full_grid
    )

    solvent_index = N - 1

    # Since solvent is constrained:
    #
    #   phi_solvent = 1 - sum_i phi_i
    #
    # the effective chemical potential for solute i is:
    #
    #   mu_i - mu_solvent

    mu_bulk = (
        np.log(solutes)
        - np.log(solvent)[None, :, :]
        + interaction_derivative[:n_solutes]
        - interaction_derivative[solvent_index][None, :, :]
    )

    return mu_bulk


def make_display_grid(solutes):
    """
    Returns all components including solvent as the last component.

    Output shape:
        N x cx x cy

    where:
        N - 1 solute components
        last component is solvent
    """

    solutes, solvent = enforce_bounds(solutes)

    full_grid = np.concatenate(
        [
            solutes,
            solvent[None, :, :]
        ],
        axis=0
    )

    return full_grid


def simulate_frames(
    cx=128,
    cy=128,
    N=3,
    step=0.1,
    t_max=5.0,
    kappa=5.0,
    spacing=1.0,
    mobility=None,
    A=4.0,
    beta=0.4,
    iterate=10,
):


    if N < 2:
        raise ValueError("N must be at least 2: one solute plus one solvent.")

    n_solutes = N - 1

    if mobility is None:
        print("yay")
        mobility = np.ones(n_solutes)

    mobility = np.asarray(mobility)

    if len(mobility) != n_solutes:
        mobility = np.ones(n_solutes)

    components = np.ones(n_solutes) * beta / n_solutes

    solutes = np.array([
        c * np.ones((cx, cy)) + np.random.normal(0, 0.1, (cx, cy))
        for c in components
    ])

    # Optional seed.
    # solutes[0][40:60, 40:60] = 0.1
    # solutes[1][40:60, 40:60] = 0.5

    solutes, solvent = enforce_bounds(solutes)

    # ------------------------------------------------------------
    # Full chi matrix including solvent.
    #
    # For N = 3:
    #   component 0 = solute 0
    #   component 1 = solute 1
    #   component 2 = solvent
    # ------------------------------------------------------------

    chi = np.zeros((N, N))

    # Example: solute 0 repels solvent.
    chi[0, -1] = 4
    chi[-1, 0] = 4

    chi[1, -1] = 4
    chi[-1, 1] = 4

    # Optional examples:
    # solute 1 repels solvent.
    # chi[1, 0] = -10
    # chi[0, 1] = -10

    # solute 0 attracts solute 1.
    chi[0, 1] = 3.5
    chi[1, 0] = 3.5

    kx = 2 * np.pi * np.fft.fftfreq(cx, d=spacing)
    ky = 2 * np.pi * np.fft.fftfreq(cy, d=spacing)

    qx, qy = np.meshgrid(kx, ky, indexing="ij")

    q2 = qx**2 + qy**2
    q4 = q2**2

    time = 0.0
    grid = make_display_grid(solutes).tolist()

    # Send initial frame.
    yield {
        "time": 0.0,
        "grid": make_display_grid(solutes).tolist(),
    }

    for t in np.arange(step, t_max + step, step):
        solutes, solvent = enforce_bounds(solutes)

        fft_grid = np.fft.fft2(solutes, axes=(1, 2))

        mu_bulk = chemical_potential_bulk(solutes, chi)
        mu_bulk_hat = np.fft.fft2(mu_bulk, axes=(1, 2))

        norm = np.zeros(solvent.shape, dtype=np.complex128)

        for i in range(n_solutes):

            pot = mu_bulk_hat[i] + kappa * q2 * fft_grid[i]

            nl = (
                -q2 * pot
                + A * q4 * fft_grid[i]
            ) * mobility[i]

            incr = (
                fft_grid[i] + step * nl
            ) / (
                1.0 + A * mobility[i] * q4 * step
            )

            norm += incr - fft_grid[i]
            fft_grid[i] = incr

        # Your original approximate correction.
        fft_grid -= norm / (len(solutes) + 1)

        solutes = np.real(
            np.fft.ifft2(fft_grid, axes=(1, 2))
        )

        solutes, solvent = enforce_bounds(solutes)

        if t // step % iterate == 0:
            time = float(t)
            grid = make_display_grid(solutes).tolist()

        yield {
            "time": time,
            "grid": grid,
        }
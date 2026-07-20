from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

volNs = 4.35e-21
Nav = 6.02214076e23

R = 1.987204258e-3  # kcal / (mol K)

DELTA_H_STICKY = -27.0       # kcal/mol
DELTA_S_STICKY = -0.070       # kcal/(mol K)

WORKSPACE = Path(__file__).resolve().parent
CSV_FILE_NAME = "binodal_results_betaDeltaG_corrected.csv"
CSV_PATH = WORKSPACE / CSV_FILE_NAME


def load_binodal_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is missing a header row")

        required = {"beta_Delta_G", "phi_minus", "phi_plus"}
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_list}")

        beta = []
        phi_dilute = []
        phi_dense = []

        for row in reader:
            beta.append(float(row["beta_Delta_G"]))
            temperature_K = DELTA_H_STICKY / (
                R * float(row["beta_Delta_G"]) + DELTA_S_STICKY
            )
            print(temperature_K)
            phi_dilute.append(float(row["phi_minus"])/(volNs*Nav) * 10**6)
            phi_dense.append(float(row["phi_plus"])/(volNs*Nav)*10**6)

    return np.asarray(beta), np.asarray(phi_dilute), np.asarray(phi_dense)


def plot_binodal(csv_path: Path) -> None:
    beta, phi_dilute, phi_dense = load_binodal_data(csv_path)

    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    plt.xscale("log")

    ax.fill_betweenx(beta, phi_dilute, phi_dense, alpha=0.2, color="tab:blue")
    ax.plot(phi_dilute, beta, color="tab:blue", linewidth=2.0, label=r"dilute branch $\phi_-$")
    ax.plot(phi_dense, beta, color="tab:red", linewidth=2.0, label=r"dense branch $\phi_+$")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\beta \Delta G$")
    ax.set_title("Binodal from CSV")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    plt.show()

    print(f"Loaded data from {csv_path.resolve()}")


def main() -> None:
    csv_path = CSV_PATH
    if not csv_path.exists():
        preferred = [
            path for path in WORKSPACE.glob("*.csv")
            if "binodal" in path.name.lower()
        ]
        if preferred:
            csv_path = sorted(preferred)[-1]
        else:
            raise FileNotFoundError("No CSV file found in the workspace")

    plot_binodal(csv_path)


if __name__ == "__main__":
    main()

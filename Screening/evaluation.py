from __future__ import annotations
import argparse
from itertools import combinations_with_replacement
from pathlib import Path
import sys
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import ViennaRNA as RNA

RNA.params_load_DNA_Mathews2004()


SURF_DIRECTORY = Path(__file__).resolve().parents[1] / "SURF2026_test"
if str(SURF_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SURF_DIRECTORY))

from orthogonal import directional_binding_delta_g, fold, reverse_complement  # noqa: E402


DEFAULT_TEMPERATURE_C = 20.0
DEFAULT_SALT_M = 1.0
GAS_CONSTANT_KCAL_PER_MOL_K = 0.00198720425864083


def thermal_binding_metrics(
    reactant_a: str,
    reactant_b: str,
    *,
    concentration_M: float = 1e-6,
    target_temperature_C: float = 30.0,
    fit_temperatures_C: tuple[float, ...] = (20.0, 25.0, 30.0, 35.0, 40.0),
    salt_M: float = DEFAULT_SALT_M,
) -> dict[str, object]:
    """Fit Delta H/S and predict occupancy/T50 for a bimolecular reaction."""
    if concentration_M <= 0:
        raise ValueError("concentration_M must be positive.")
    if len(fit_temperatures_C) < 2:
        raise ValueError("At least two fit temperatures are required.")

    def occupancy(delta_g: float, temperature_K: float) -> tuple[float, float]:
        equilibrium_constant = float(
            np.exp(
                -delta_g
                / (GAS_CONSTANT_KCAL_PER_MOL_K * temperature_K)
            )
        )
        scaled_constant = equilibrium_constant * concentration_M
        if scaled_constant <= 1e-12:
            bound = float(scaled_constant)
        else:
            bound = float(
                (
                    2.0 * scaled_constant
                    + 1.0
                    - np.sqrt(4.0 * scaled_constant + 1.0)
                )
                / (2.0 * scaled_constant)
            )
        return bound, 1.0 - bound

    curve = []
    for temperature_C in fit_temperatures_C:
        energy_a = fold(reactant_a, temperature_C, salt_M)[1]
        energy_b = fold(reactant_b, temperature_C, salt_M)[1]
        complex_energy = fold(
            f"{reactant_a}&{reactant_b}", temperature_C, salt_M
        )[1]
        delta_g = float(complex_energy - energy_a - energy_b)
        bound_fraction, off_fraction = occupancy(
            delta_g, float(temperature_C) + 273.15
        )
        curve.append(
            {
                "temperature_C": float(temperature_C),
                "delta_g_kcal_per_mol": delta_g,
                "bound_fraction": bound_fraction,
                "off_fraction": off_fraction,
            }
        )

    temperatures_K = np.array(
        [point["temperature_C"] + 273.15 for point in curve]
    )
    delta_g_values = np.array(
        [point["delta_g_kcal_per_mol"] for point in curve]
    )
    slope, intercept = np.polyfit(temperatures_K, delta_g_values, 1)
    enthalpy = float(intercept)
    entropy = float(-slope)

    target_temperature_K = target_temperature_C + 273.15
    target_delta_g = enthalpy - target_temperature_K * entropy
    bound_fraction, off_fraction = occupancy(
        float(target_delta_g), target_temperature_K
    )

    half_bound_K = 2.0 / concentration_M
    denominator = entropy - GAS_CONSTANT_KCAL_PER_MOL_K * np.log(half_bound_K)
    t50_K = enthalpy / denominator if denominator != 0 else float("nan")
    return {
        "concentration_M": concentration_M,
        "target_temperature_C": target_temperature_C,
        "curve": curve,
        "enthalpy_kcal_per_mol": enthalpy,
        "entropy_kcal_per_mol_K": entropy,
        "delta_g_at_target_kcal_per_mol": float(target_delta_g),
        "bound_fraction_at_target": bound_fraction,
        "off_fraction_at_target": off_fraction,
        "predicted_T50_C": float(t50_K - 273.15),
    }


def _model(temperature_C: float, salt_M: float) -> RNA.md:
    md = RNA.md()
    md.temperature = float(temperature_C)
    md.salt = float(salt_M)
    md.dangles = 2
    return md


def partition_function_binding_delta_g(
    sequence_a: str,
    sequence_b: str,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> float:
    """Return ensemble binding Delta G using the notebook's subtraction."""
    md = _model(temperature_C, salt_M)
    complex_energy = RNA.fold_compound(f"{sequence_a}&{sequence_b}", md).pf()[1]
    energy_a = RNA.fold_compound(sequence_a, md).pf()[1]
    energy_b = RNA.fold_compound(sequence_b, md).pf()[1]
    return float(complex_energy - energy_a - energy_b)


def partition_function_binding_matrix(
    sequences: list[str],
    *,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> np.ndarray:
    """Return the symmetric ensemble-binding matrix used by the notebook."""
    size = len(sequences)
    matrix = np.empty((size, size), dtype=float)
    for row, column in combinations_with_replacement(range(size), 2):
        value = partition_function_binding_delta_g(
            sequences[row], sequences[column], temperature_C, salt_M
        )
        matrix[row, column] = value
        matrix[column, row] = value
    return matrix


def _repeat_counts(sequence: str, max_run: int = 6) -> list[int]:
    """Count nucleotide, purine/pyrimidine, and weak/strong runs."""
    encodings = (
        sequence,
        sequence.translate(str.maketrans({"A": "R", "G": "R", "T": "P", "C": "P"})),
        sequence.translate(str.maketrans({"A": "W", "T": "W", "G": "S", "C": "S"})),
    )
    counts = [0] * (max_run + 1)
    for encoded in encodings:
        run = 1
        for previous, current in zip(encoded, encoded[1:]):
            if current == previous:
                run += 1
            else:
                counts[min(run, max_run)] += 1
                run = 1
        counts[min(run, max_run)] += 1
    return counts


def _shared_subsequence_counts(
    sequences: list[str], min_length: int = 3, max_length: int = 8
) -> dict[int, int]:
    """Count repeated system-wide subsequences as in notebook system_SSM."""
    present: set[str] = set()
    counts = {length: 0 for length in range(min_length, max_length + 1)}
    for sequence in sequences:
        for length in counts:
            for start in range(len(sequence) - length + 1):
                subsequence = sequence[start : start + length]
                if subsequence in present:
                    counts[length] += 1
                else:
                    present.add(subsequence)
    return counts


def evaluate_domain_effectiveness(
    domain_a: str,
    domain_b: str,
    domain_c: str,
    domain_d: str,
    *,
    intended_target_kcal_per_mol: float = -8.0,
    off_target_kcal_per_mol: float = 0.0,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> dict[str, object]:
    """Assess A-D using metrics adapted from the design notebook.

    The central metric is the partition-function binding matrix. Intended
    entries are A-A*, B-B*, C-C*, and D-D*; every other entry, including
    homodimers, is treated as off-target.
    """
    domains = [domain_a.upper(), domain_b.upper(), domain_c.upper(), domain_d.upper()]
    if any(not domain or set(domain) - set("ATCG") for domain in domains):
        raise ValueError("Domains must be nonempty DNA sequences using A, T, C, G.")
    sequences = domains + [reverse_complement(domain) for domain in domains]
    labels = ["A", "B", "C", "D", "A*", "B*", "C*", "D*"]
    matrix = partition_function_binding_matrix(
        sequences, temperature_C=temperature_C, salt_M=salt_M
    )

    intended_pairs = [(index, index + 4) for index in range(4)]
    intended_values = np.array([matrix[i, j] for i, j in intended_pairs])
    intended_set = {tuple(sorted(pair)) for pair in intended_pairs}
    off_target_entries = [
        (row, column, matrix[row, column])
        for row, column in combinations_with_replacement(range(8), 2)
        if (row, column) not in intended_set
    ]
    off_target_values = np.array([entry[2] for entry in off_target_entries])
    worst_row, worst_column, worst_value = min(
        off_target_entries, key=lambda entry: entry[2]
    )

    target_matrix = np.full((8, 8), float(off_target_kcal_per_mol))
    for first, second in intended_pairs:
        target_matrix[first, second] = intended_target_kcal_per_mol
        target_matrix[second, first] = intended_target_kcal_per_mol

    selectivity_margins = {}
    for index, label in enumerate(labels[:4]):
        cognate = matrix[index, index + 4]
        strongest_unintended = min(
            matrix[index, other]
            for other in range(8)
            if other != index + 4
        )
        selectivity_margins[label] = float(strongest_unintended - cognate)

    monomer_structures = {}
    monomer_mfes = {}
    for label, sequence in zip(labels, sequences):
        structure, energy = RNA.fold_compound(
            sequence, _model(temperature_C, salt_M)
        ).mfe()
        monomer_structures[label] = structure
        monomer_mfes[label] = float(energy)

    composition = {
        label: {base: sequence.count(base) / len(sequence) for base in "ATCG"}
        for label, sequence in zip(labels[:4], domains)
    }
    repeat_counts = np.sum(
        np.array([_repeat_counts(sequence) for sequence in domains]), axis=0
    )
    ssm_counts = _shared_subsequence_counts(domains)

    return {
        "labels": labels,
        "sequences": sequences,
        "partition_function_matrix_kcal_per_mol": matrix,
        "target_matrix_kcal_per_mol": target_matrix,
        "matrix_rmse_kcal_per_mol": float(np.sqrt(np.mean((matrix - target_matrix) ** 2))),
        "intended_binding_kcal_per_mol": {
            label: float(value) for label, value in zip(labels[:4], intended_values)
        },
        "intended_mean_kcal_per_mol": float(np.mean(intended_values)),
        "intended_std_kcal_per_mol": float(np.std(intended_values)),
        "intended_target_rmse_kcal_per_mol": float(
            np.sqrt(np.mean((intended_values - intended_target_kcal_per_mol) ** 2))
        ),
        "worst_off_target": {
            "pair": (labels[worst_row], labels[worst_column]),
            "delta_g_kcal_per_mol": float(worst_value),
        },
        "off_target_mean_kcal_per_mol": float(np.mean(off_target_values)),
        "off_target_rms_kcal_per_mol": float(
            np.sqrt(np.mean((off_target_values - off_target_kcal_per_mol) ** 2))
        ),
        "selectivity_margins_kcal_per_mol": selectivity_margins,
        "minimum_selectivity_margin_kcal_per_mol": min(selectivity_margins.values()),
        "monomer_structures": monomer_structures,
        "monomer_mfes_kcal_per_mol": monomer_mfes,
        "all_monomers_unstructured": all(
            structure == "." * len(sequence)
            for structure, sequence in zip(monomer_structures.values(), sequences)
        ),
        "base_composition": composition,
        "worst_base_fraction_deviation_from_0.25": max(
            abs(fraction - 0.25)
            for frequencies in composition.values()
            for fraction in frequencies.values()
        ),
        "repeat_counts_3_to_6": {
            length: int(repeat_counts[length]) for length in range(3, 7)
        },
        "shared_subsequence_clashes_3_to_8": ssm_counts,
        "contains_GGGG": any("GGGG" in sequence for sequence in domains),
    }


def domain_orthogonality_matrix(
    domain_a: str,
    domain_b: str,
    domain_c: str,
    domain_d: str,
    *,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> tuple[list[str], np.ndarray]:
    """Return the 8x8 interaction matrix for A-D and A*-D*.

    Values are effective binding Delta G in kcal/mol. More-negative values
    indicate stronger binding. The intended cognate pairs A-A*, B-B*, C-C*,
    and D-D* therefore appear as strong interactions in the matrix.
    """
    domains = [
        domain_a.upper(),
        domain_b.upper(),
        domain_c.upper(),
        domain_d.upper(),
    ]
    if any(not domain or set(domain) - set("ATCG") for domain in domains):
        raise ValueError("Domains must be nonempty DNA sequences using A, T, C, G.")

    sequences = domains + [reverse_complement(domain) for domain in domains]
    labels = ["A", "B", "C", "D", "A*", "B*", "C*", "D*"]
    matrix = np.empty((8, 8), dtype=float)
    for row, column in combinations_with_replacement(range(8), 2):
        delta_g = directional_binding_delta_g(
            sequences[row],
            sequences[column],
            temperature_C,
            salt_M,
        )
        matrix[row, column] = delta_g
        matrix[column, row] = delta_g
    return labels, matrix


def matrix_from_screening_result(
    result: Mapping[str, object],
    *,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> tuple[list[str], np.ndarray]:
    """Build the matrix directly from one completed screening result."""
    required = ("toehold_A", "toehold_B", "domain_C", "domain_D")
    missing = [key for key in required if not result.get(key)]
    if missing:
        raise ValueError(f"Screening result is missing: {', '.join(missing)}")
    return domain_orthogonality_matrix(
        *(str(result[key]) for key in required),
        temperature_C=temperature_C,
        salt_M=salt_M,
    )


def effectiveness_from_screening_result(
    result: Mapping[str, object],
    **kwargs: object,
) -> dict[str, object]:
    """Evaluate all notebook-derived metrics for one completed screen result."""
    required = ("toehold_A", "toehold_B", "domain_C", "domain_D")
    missing = [key for key in required if not result.get(key)]
    if missing:
        raise ValueError(f"Screening result is missing: {', '.join(missing)}")
    return evaluate_domain_effectiveness(
        *(str(result[key]) for key in required),
        **kwargs,
    )


def plot_orthogonality_matrix(
    labels: list[str],
    matrix: np.ndarray,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the domain matrix and optionally save it as an image."""
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="viridis", aspect="equal")
    figure.colorbar(image, ax=axis, label="Effective binding ΔG (kcal/mol)")
    axis.set_xticks(range(8), labels=labels)
    axis.set_yticks(range(8), labels=labels)
    axis.set_title("Domain/complement interaction matrix")

    midpoint = (float(matrix.min()) + float(matrix.max())) / 2.0
    for row in range(8):
        for column in range(8):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < midpoint else "black",
            )
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def print_effectiveness_report(report: Mapping[str, object]) -> None:
    """Print the notebook-derived effectiveness metrics compactly."""
    print("Partition-function interaction matrix (kcal/mol):")
    print(report["partition_function_matrix_kcal_per_mol"])
    print("Intended binding:", report["intended_binding_kcal_per_mol"])
    print(
        "Intended mean/std/target RMSE: "
        f"{report['intended_mean_kcal_per_mol']:.3f} / "
        f"{report['intended_std_kcal_per_mol']:.3f} / "
        f"{report['intended_target_rmse_kcal_per_mol']:.3f} kcal/mol"
    )
    print("Worst off-target:", report["worst_off_target"])
    print(
        "Off-target RMS: "
        f"{report['off_target_rms_kcal_per_mol']:.3f} kcal/mol"
    )
    print("Selectivity margins:", report["selectivity_margins_kcal_per_mol"])
    print("All monomers unstructured:", report["all_monomers_unstructured"])
    print("Run counts (3-6):", report["repeat_counts_3_to_6"])
    print(
        "Shared-subsequence clashes (3-8):",
        report["shared_subsequence_clashes_3_to_8"],
    )
    print("Contains GGGG:", report["contains_GGGG"])
    if report.get("thermal_release_metrics"):
        print("Thermal release metrics at 1 uM:")
        for name, metrics in report["thermal_release_metrics"].items():
            print(
                f"  {name}: T50={metrics['predicted_T50_C']:.2f} deg C, "
                f"off at 30 deg C={100 * metrics['off_fraction_at_target']:.1f}%, "
                f"dH={metrics['enthalpy_kcal_per_mol']:.3f} kcal/mol, "
                f"dS={metrics['entropy_kcal_per_mol_K']:.6f} kcal/mol/K"
            )
            print("    temperature curve:")
            for point in metrics["curve"]:
                print(
                    f"      {point['temperature_C']:4.0f} deg C: "
                    f"dG={point['delta_g_kcal_per_mol']:7.3f} kcal/mol, "
                    f"bound={100 * point['bound_fraction']:6.2f}%, "
                    f"off={100 * point['off_fraction']:6.2f}%"
                )
    if report.get("requested_condition_diagnostics"):
        diagnostics = report["requested_condition_diagnostics"]
        print("Requested-condition diagnostics (reported only; not screened):")
        print(
            "  First linker off at 20 deg C: "
            f"{100 * diagnostics['first_linker_off_fraction_at_20C']:.2f}% "
            f"(stable target met: "
            f"{diagnostics['first_linker_stable_at_20C_10_to_20_percent_off']})"
        )
        print(
            "  Blocker T50 values: "
            f"{diagnostics['blocker_1_T50_C']:.2f} and "
            f"{diagnostics['blocker_2_T50_C']:.2f} deg C; "
            f"separation={diagnostics['blocker_T50_separation_C']:.2f} deg C "
            f"(within 2 deg C: {diagnostics['blockers_within_2C']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", required=True, help="Domain A sequence")
    parser.add_argument("--B", required=True, help="Domain B sequence")
    parser.add_argument("--C", required=True, help="Domain C sequence")
    parser.add_argument("--D", required=True, help="Domain D sequence")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_C)
    parser.add_argument("--salt", type=float, default=DEFAULT_SALT_M)
    parser.add_argument("--intended-target", type=float, default=-8.0)
    parser.add_argument("--off-target", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    domain_labels, interaction_matrix = domain_orthogonality_matrix(
        args.A,
        args.B,
        args.C,
        args.D,
        temperature_C=args.temperature,
        salt_M=args.salt,
    )
    print("Labels:", domain_labels)
    print(interaction_matrix)
    effectiveness = evaluate_domain_effectiveness(
        args.A,
        args.B,
        args.C,
        args.D,
        intended_target_kcal_per_mol=args.intended_target,
        off_target_kcal_per_mol=args.off_target,
        temperature_C=args.temperature,
        salt_M=args.salt,
    )
    print_effectiveness_report(effectiveness)
    plot_orthogonality_matrix(
        domain_labels,
        interaction_matrix,
        output_path=args.output,
        show=not args.no_show,
    )

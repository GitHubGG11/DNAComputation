"""Run linker screening and notebook-style effectiveness evaluation together."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
from typing import Any, Sequence

try:
    from .extract_ns_arms import DEFAULT_WORKBOOK, get_ns_arms
    from .evaluation import (
        effectiveness_from_screening_result,
        plot_orthogonality_matrix,
        print_effectiveness_report,
        thermal_binding_metrics,
    )
    from .linker_screening import (
        DEFAULT_SALT_M,
        DEFAULT_BLOCKER_HEAT_C,
        DEFAULT_TEMPERATURE_C,
        DEFAULT_THRESHOLD_KCAL_PER_MOL,
        DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL,
        DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL,
        reverse_complement,
        screen_linker_toeholds,
    )
except ImportError:  # Allow ``python Screening/screening_pipeline.py``.
    from extract_ns_arms import DEFAULT_WORKBOOK, get_ns_arms
    from evaluation import (
        effectiveness_from_screening_result,
        plot_orthogonality_matrix,
        print_effectiveness_report,
        thermal_binding_metrics,
    )
    from linker_screening import (
        DEFAULT_SALT_M,
        DEFAULT_BLOCKER_HEAT_C,
        DEFAULT_TEMPERATURE_C,
        DEFAULT_THRESHOLD_KCAL_PER_MOL,
        DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL,
        DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL,
        reverse_complement,
        screen_linker_toeholds,
    )


def screen_and_evaluate(
    nanostar_a_arms: Sequence[str],
    nanostar_b_arms: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD_KCAL_PER_MOL,
    binding_target: float = DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL,
    binding_tolerance: float = DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL,
    evaluation_intended_target: float = -8.0,
    evaluation_off_target: float = 0.0,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
    heat: float = DEFAULT_BLOCKER_HEAT_C,
    matrix_directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Screen linker designs, evaluate accepted domains, and return both."""
    screened = [screen_linker_toeholds(
        nanostar_a_arms,
        nanostar_b_arms,
        threshold=threshold,
        binding_target=binding_target,
        binding_tolerance=binding_tolerance,
        temperature_C=temperature_C,
        salt_M=salt_M,
        heat=heat,
    )]

    output_directory = Path(matrix_directory) if matrix_directory else None
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, Any]] = []
    for result in screened:
        record = dict(result)
        if not result["complete"]:
            record["evaluation"] = None
            combined.append(record)
            continue

        evaluation = effectiveness_from_screening_result(
            result,
            intended_target_kcal_per_mol=evaluation_intended_target,
            off_target_kcal_per_mol=evaluation_off_target,
            temperature_C=temperature_C,
            salt_M=salt_M,
        )
        nanostar_complex = "&".join(
            [*result["nanostar_A_strands"], *result["nanostar_B_strands"]]
        )
        evaluation["thermal_release_metrics"] = {
            "first_linker": thermal_binding_metrics(
                nanostar_complex,
                result["full_linker"],
                concentration_M=1e-6,
                target_temperature_C=30.0,
                salt_M=salt_M,
            ),
            "blocker_1": thermal_binding_metrics(
                result["blocker_1"],
                reverse_complement(result["blocker_1"]),
                concentration_M=1e-6,
                target_temperature_C=30.0,
                salt_M=salt_M,
            ),
            "blocker_2": thermal_binding_metrics(
                result["blocker_2"],
                reverse_complement(result["blocker_2"]),
                concentration_M=1e-6,
                target_temperature_C=30.0,
                salt_M=salt_M,
            ),
        }
        for metrics in evaluation["thermal_release_metrics"].values():
            metrics["T50_target_C"] = 30.0
            metrics["T50_error_C"] = metrics["predicted_T50_C"] - 30.0
            metrics["meets_T50_target_within_2C"] = (
                abs(metrics["T50_error_C"]) <= 2.0
            )
        evaluation["all_thermal_release_targets_met"] = all(
            metrics["meets_T50_target_within_2C"]
            for metrics in evaluation["thermal_release_metrics"].values()
        )
        first_linker_20 = next(
            point
            for point in evaluation["thermal_release_metrics"]["first_linker"]["curve"]
            if point["temperature_C"] == 20.0
        )
        blocker_1_t50 = evaluation["thermal_release_metrics"]["blocker_1"][
            "predicted_T50_C"
        ]
        blocker_2_t50 = evaluation["thermal_release_metrics"]["blocker_2"][
            "predicted_T50_C"
        ]
        evaluation["requested_condition_diagnostics"] = {
            "first_linker_off_fraction_at_20C": first_linker_20["off_fraction"],
            "first_linker_stable_at_20C_10_to_20_percent_off": (
                first_linker_20["off_fraction"] <= 0.20
            ),
            "blocker_1_T50_C": blocker_1_t50,
            "blocker_2_T50_C": blocker_2_t50,
            "blocker_T50_separation_C": abs(blocker_1_t50 - blocker_2_t50),
            "blockers_within_2C": abs(blocker_1_t50 - blocker_2_t50) <= 2.0,
        }
        record["evaluation"] = evaluation
        combined.append(record)

        print(
            "\nEffectiveness for nanostars A and B:"
        )
        print_effectiveness_report(evaluation)

        if output_directory is not None:
            filename = "nanostar_A_B_matrix.png"
            plot_orthogonality_matrix(
                evaluation["labels"],
                evaluation["partition_function_matrix_kcal_per_mol"],
                output_path=output_directory / filename,
                show=False,
            )

    return combined


def save_pipeline_report(
    results: list[dict[str, Any]],
    report_file: str | Path,
) -> Path:
    """Save domain sequences, complete constructs, and all metrics as text."""
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "NANOSTAR LINKER SCREENING AND EFFECTIVENESS REPORT",
        "=" * 52,
        f"Designs: {len(results)}",
        f"Completed: {sum(result['complete'] for result in results)}",
        "",
    ]

    for result in results:
        lines.extend(
            [
                "NANOSTAR A + NANOSTAR B",
                "-" * 52,
                f"Complete: {result['complete']}",
            ]
        )
        if not result["complete"]:
            lines.extend(["No complete design found.", ""])
            continue

        lines.extend(
            [
                "",
                "DOMAINS",
                f"A:  {result['toehold_A']}",
                f"A*: {result['toehold_A_rc']}",
                f"B:  {result['toehold_B']}",
                f"B*: {result['toehold_B_rc']}",
                f"C:  {result['domain_C']}",
                f"C*: {result['domain_C_rc']}",
                f"D:  {result['domain_D']}",
                f"D*: {result['domain_D_rc']}",
                "",
                "FULL NANOSTAR A STRANDS",
            ]
        )
        lines.extend(
            f"A{index}: {strand}"
            for index, strand in enumerate(result["nanostar_A_strands"], start=1)
        )
        lines.append("")
        lines.append("FULL NANOSTAR B STRANDS")
        lines.extend(
            f"B{index}: {strand}"
            for index, strand in enumerate(result["nanostar_B_strands"], start=1)
        )
        lines.extend(
            [
                "",
                "FULL LINKERS",
                f"Linker 1 (A*-C & C*-B*): {result['full_linker']}",
                f"Linker 2 (B*-D-B*): {result['linker_Bstar_D_Bstar']}",
                f"Linker 3 (A*-D-A*): {result['linker_Astar_D_Astar']}",
                "",
                "BLOCKERS",
                f"Blocker 1: {result['blocker_1']}",
                f"Blocker 2: {result['blocker_2']}",
                "",
                "ALL SCREENING DATA",
                pformat(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {
                            "evaluation",
                            "nanostar_A_strands",
                            "nanostar_B_strands",
                        }
                    },
                    sort_dicts=False,
                    width=140,
                ),
                "",
                "ALL EFFECTIVENESS METRICS",
                pformat(result["evaluation"], sort_dicts=False, width=140),
                "",
                "=" * 52,
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_KCAL_PER_MOL)
    parser.add_argument(
        "--binding-target",
        type=float,
        default=DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL,
    )
    parser.add_argument(
        "--binding-tolerance",
        type=float,
        default=DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL,
    )
    parser.add_argument("--evaluation-target", type=float, default=-8.0)
    parser.add_argument("--evaluation-off-target", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_C)
    parser.add_argument("--salt", type=float, default=DEFAULT_SALT_M)
    parser.add_argument(
        "--heat",
        type=float,
        default=DEFAULT_BLOCKER_HEAT_C,
        help="Temperature in deg C at which at least 50%% of each blocker must be dissociated.",
    )
    parser.add_argument(
        "--matrix-directory",
        type=Path,
        default=None,
        help="Optional directory for accepted-bucket heatmaps.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path(__file__).resolve().parent / "screening_metrics.txt",
        help="Text report path (default: Screening/screening_metrics.txt).",
    )
    args = parser.parse_args()

    nanostar_A_arms = (
        "GCGTCCGACACTGAACTATG", "TGTCAGCCGTGCTATCAAGA",
        "ACGGTCTGACCCGAAATAGT", "ATGTGGCCTACAGTGAATCC",
    )
    nanostar_B_arms = (
        "TTGACCACCTAGGATGCGTT", "CGGAGACTAGATGATTTCCG",
        "GATGTCTAACGATTCAGGCC", "CAAGTATCGGTGCTGATCCA",
    )
    results = screen_and_evaluate(
        nanostar_A_arms,
        nanostar_B_arms,
        threshold=args.threshold,
        binding_target=args.binding_target,
        binding_tolerance=args.binding_tolerance,
        evaluation_intended_target=args.evaluation_target,
        evaluation_off_target=args.evaluation_off_target,
        temperature_C=args.temperature,
        salt_M=args.salt,
        heat=args.heat,
        matrix_directory=args.matrix_directory,
    )
    completed = sum(result["complete"] for result in results)
    evaluated = sum(result["evaluation"] is not None for result in results)
    report_path = save_pipeline_report(results, args.report_file)
    print(
        f"\nPipeline complete: {completed}/{len(results)} screened buckets; "
        f"{evaluated} evaluated."
    )
    print(f"Text report saved to: {report_path}")

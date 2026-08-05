"""Screen orthogonal linker toeholds for all 16-choose-2 nanostar pairs."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from itertools import combinations
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import ViennaRNA as RNA

try:
    from .extract_ns_arms import DEFAULT_WORKBOOK, get_ns_arms
except ImportError:
    from extract_ns_arms import DEFAULT_WORKBOOK, get_ns_arms


SURF_DIRECTORY = Path(__file__).resolve().parents[1] / "SURF2026_test"
if str(SURF_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SURF_DIRECTORY))

from orthogonal import (  # noqa: E402
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_SALT_M,
    DEFAULT_TEMPERATURE_C,
    DEFAULT_THRESHOLD_KCAL_PER_MOL,
    directional_binding_delta_g,
    fold,
    grow_orthogonal_buckets,
    interaction_delta_g,
    next_unstructured_sequence,
    register_unstructured_sequence,
    reverse_complement,
    self_interaction_delta_g,
)


NANOSTAR_COUNT = 16
TOEHOLD_LENGTH = 8
THIRD_DOMAIN_LENGTH = 6
D_DOMAIN_LENGTH = 8
DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL = -8.0
DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL = 1.0


def _nanostar_strands_with_toehold(
    nanostar_number: int,
    toehold: str,
    workbook: str | Path = DEFAULT_WORKBOOK,
) -> tuple[str, str, str, str]:
    """Build the four complete strands for one numbered nanostar.

    Arms are read in workbook order. Each strand contains the reverse
    complement of the preceding arm, ``TT``, its own arm, ``T``, and the same
    exposed A or B toehold. For strand 1, the preceding arm wraps around to arm
    4. The returned tuple is ordered strand 1 through strand 4.
    """
    # Four consecutive workbook arms define one nanostar.
    arms = get_ns_arms(workbook)
    start = (nanostar_number - 1) * 4
    group = arms[start : start + 4]
    if len(group) != 4:
        raise ValueError(f"Nanostar {nanostar_number} does not have four arms.")
    # Cyclic construction: strand n starts with RC(arm n-1).
    return tuple(
        reverse_complement(group[index - 1])
        + "TT"
        + group[index]
        + "T"
        + toehold
        for index in range(4)
    )


def _screen_domain_d_and_blockers(
    toehold_a: str,
    toehold_b: str,
    domain_c: str,
    *,
    threshold: float,
    temperature_C: float,
    salt_M: float,
    max_candidates: int,
) -> dict[str, Any] | None:
    """Find domain D, two D-linkers, and two correctly registered blockers.

    D is an unstructured 8-mer that must avoid self-binding and unintended
    binding to A, B, or C. It is used to construct ``B*-D-B*`` and
    ``A*-D-A*``; both complete strands must remain unstructured in isolation.

    Two blockers are then designed against the two A*/D junctions of
    ``A*-D-A*``. Each covers four D bases and three to five A* bases. A design
    is returned only when the blockers are unstructured, mutually orthogonal,
    and their MFE structures bind the full linker at the exact intended
    registers (no shifted/slipped duplex). Returns ``None`` if the search is
    exhausted.
    """
    # Cache monomer energies needed by orthogonal.py's interaction functions.
    rng = random.Random()
    seen: set[str] = set()
    monomer_mfes: dict[str, float] = {}
    reference_domains = (toehold_a, toehold_b, domain_c)
    for domain in reference_domains:
        if not register_unstructured_sequence(
            domain, monomer_mfes, temperature_C, salt_M
        ):
            return None

    a_star = reverse_complement(toehold_a)
    b_star = reverse_complement(toehold_b)
    md = RNA.md()
    md.temperature = float(temperature_C)
    md.salt = float(salt_M)
    md.dangles = 2

    # Search random unstructured D candidates until all downstream constructs
    # pass; a failed linker or blocker sends the search to the next D.
    for evaluated_count in range(1, max_candidates + 1):
        try:
            domain_d = next_unstructured_sequence(
                rng,
                D_DOMAIN_LENGTH,
                seen,
                monomer_mfes,
                temperature_C,
                salt_M,
            )
        except RuntimeError:
            return None

        # D must avoid homodimers and every unintended orientation with A/B/C.
        self_energy = self_interaction_delta_g(
            domain_d, monomer_mfes, temperature_C, salt_M
        )
        cross_energies = {
            name: interaction_delta_g(
                domain_d,
                domain,
                monomer_mfes,
                temperature_C,
                salt_M,
            )
            for name, domain in zip(("D-A", "D-B", "D-C"), reference_domains)
        }
        if min(self_energy, *cross_energies.values()) < threshold:
            continue

        # These are the two single-strand D-containing linker designs.
        linker_bdb = b_star + domain_d + b_star
        linker_ada = a_star + domain_d + a_star
        linker_structures = {
            "B*-D-B*": fold(linker_bdb, temperature_C, salt_M)[0],
            "A*-D-A*": fold(linker_ada, temperature_C, salt_M)[0],
        }
        if any(
            structure != "." * len(sequence)
            for structure, sequence in zip(
                linker_structures.values(), (linker_bdb, linker_ada)
            )
        ):
            continue

        # Try each allowed A* coverage length independently at both junctions.
        for left_overlap in range(3, 6):
            left_target = a_star[-left_overlap:] + domain_d[:4]
            blocker_1 = reverse_complement(left_target)
            if not register_unstructured_sequence(
                blocker_1, monomer_mfes, temperature_C, salt_M
            ):
                continue
            for right_overlap in range(3, 6):
                right_target = domain_d[-4:] + a_star[:right_overlap]
                blocker_2 = reverse_complement(right_target)
                if not register_unstructured_sequence(
                    blocker_2, monomer_mfes, temperature_C, salt_M
                ):
                    continue

                # Released blockers should not bind one another strongly.
                blocker_pair_energy = interaction_delta_g(
                    blocker_1,
                    blocker_2,
                    monomer_mfes,
                    temperature_C,
                    salt_M,
                )
                if blocker_pair_energy < threshold:
                    continue

                # Fold each blocker against the entire A*-D-A* strand. The
                # exact dot-bracket register must be the MFE, preventing a
                # blocker from sliding to an alternative A*/D alignment.
                blocker_checks = {}
                for name, blocker, target_start in (
                    ("blocker_1", blocker_1, len(a_star) - left_overlap),
                    ("blocker_2", blocker_2, len(a_star) + len(domain_d) - 4),
                ):
                    target_length = len(blocker)
                    intended = (
                        "." * target_start
                        + "(" * target_length
                        + "." * (len(linker_ada) - target_start - target_length)
                        + ")" * target_length
                    )
                    compound = RNA.fold_compound(f"{linker_ada}&{blocker}", md)
                    blocker_mfe_structure, blocker_mfe = compound.mfe()
                    intended_energy = float(compound.eval_structure(intended))
                    blocker_checks[name] = {
                        "mfe_structure": blocker_mfe_structure,
                        "intended_structure": intended,
                        "mfe_kcal_per_mol": float(blocker_mfe),
                        "intended_energy_kcal_per_mol": intended_energy,
                        "slippage_free": blocker_mfe_structure == intended,
                    }
                if not all(
                    check["slippage_free"] for check in blocker_checks.values()
                ):
                    continue

                blocker_target_energies = {
                    "blocker_1-target": directional_binding_delta_g(
                        blocker_1, left_target, temperature_C, salt_M
                    ),
                    "blocker_2-target": directional_binding_delta_g(
                        blocker_2, right_target, temperature_C, salt_M
                    ),
                }
                return {
                    "domain_D": domain_d,
                    "domain_D_rc": reverse_complement(domain_d),
                    "linker_Bstar_D_Bstar": linker_bdb,
                    "linker_Astar_D_Astar": linker_ada,
                    "D_linker_structures": linker_structures,
                    "blocker_1": blocker_1,
                    "blocker_2": blocker_2,
                    "blocker_1_Astar_overlap": left_overlap,
                    "blocker_2_Astar_overlap": right_overlap,
                    "blocker_pair_delta_g_kcal_per_mol": blocker_pair_energy,
                    "blocker_target_binding_kcal_per_mol": blocker_target_energies,
                    "blocker_slippage_checks": blocker_checks,
                    "blockers_slippage_free": True,
                    "domain_D_self_delta_g_kcal_per_mol": self_energy,
                    "domain_D_cross_delta_g_kcal_per_mol": cross_energies,
                    "domain_D_evaluated_candidates": evaluated_count,
                }
    return None


def _screen_toehold_pair(
    *,
    threshold: float,
    binding_target: float,
    binding_tolerance: float,
    temperature_C: float,
    salt_M: float,
    max_candidates: int,
    workers: int | None,
) -> tuple[list[str], dict[tuple[str, str], float], int, dict[str, float]]:
    """Find length-8 A and B domains with the requested binding behavior.

    ``grow_orthogonal_buckets`` supplies two domains for which A, B, A*, and
    B* are unstructured and unintended interactions meet ``threshold``. This
    wrapper additionally requires both intended A-A* and B-B* MFE binding
    energies to lie within ``binding_tolerance`` of ``binding_target``.
    Searches are restarted until a pair passes or the candidate budget is
    exhausted. It returns sequences, cached interaction energies, the number
    evaluated, and the two cognate binding energies.
    """
    evaluated_total = 0
    while evaluated_total < max_candidates:
        remaining = max_candidates - evaluated_total
        # orthogonal.py prints every intermediate bucket; keep the 120-bucket
        # linker run readable and report only accepted linker designs below.
        with redirect_stdout(io.StringIO()):
            buckets, result_index, energies, evaluated = grow_orthogonal_buckets(
                bucket_count=1,
                clique_size=2,
                length=TOEHOLD_LENGTH,
                threshold=threshold,
                temperature_C=temperature_C,
                salt_M=salt_M,
                max_candidates=remaining,
                workers=workers,
            )
        evaluated_total += evaluated
        sequences = buckets[result_index]
        if len(sequences) < 2:
            return sequences, energies, evaluated_total, {}

        # Orthogonality deliberately excludes these intended cognate duplexes,
        # so evaluate their strengths separately here.
        binding_energies = {
            "A-A*": directional_binding_delta_g(
                sequences[0],
                reverse_complement(sequences[0]),
                temperature_C,
                salt_M,
            ),
            "B-B*": directional_binding_delta_g(
                sequences[1],
                reverse_complement(sequences[1]),
                temperature_C,
                salt_M,
            ),
        }
        if all(
            abs(energy - binding_target) <= binding_tolerance
            for energy in binding_energies.values()
        ):
            return sequences, energies, evaluated_total, binding_energies

    return [], {}, evaluated_total, {}


def _screen_third_domain(
    toehold_a: str,
    toehold_b: str,
    *,
    threshold: float,
    temperature_C: float,
    salt_M: float,
    max_candidates: int,
) -> dict[str, Any] | None:
    """Find the 6-mer C domain and construct the two-strand first linker.

    C and C* must be unstructured alone, resist homodimers, and avoid
    unintended interactions with A/B and their complements. The resulting
    linker is ``A*-C & C*-B*``. Its exact MFE structure must contain only the
    intended C-C* duplex, leaving A* and B* exposed. Exact structure equality,
    rather than energy equality alone, rejects slipped binding registers.
    """
    # Register A/B so C can be compared with both orientations of each domain.
    rng = random.Random()
    seen: set[str] = set()
    monomer_mfes: dict[str, float] = {}
    for toehold in (toehold_a, toehold_b):
        if not register_unstructured_sequence(
            toehold, monomer_mfes, temperature_C, salt_M
        ):
            return None

    for evaluated_count in range(1, max_candidates + 1):
        try:
            domain_c = next_unstructured_sequence(
                rng,
                THIRD_DOMAIN_LENGTH,
                seen,
                monomer_mfes,
                temperature_C,
                salt_M,
            )
        except RuntimeError:
            return None
        # Reject C if it self-associates or cross-binds A/B too strongly.
        self_energy = self_interaction_delta_g(
            domain_c, monomer_mfes, temperature_C, salt_M
        )
        cross_energies = {
            name: interaction_delta_g(
                domain_c,
                toehold,
                monomer_mfes,
                temperature_C,
                salt_M,
            )
            for name, toehold in (("C-A", toehold_a), ("C-B", toehold_b))
        }
        if min(self_energy, *cross_energies.values()) < threshold:
            continue

        # Strand 1 presents A*; strand 2 presents B*. C-C* holds them together.
        domain_c_rc = reverse_complement(domain_c)
        linker_1 = reverse_complement(toehold_a) + domain_c
        linker_2 = domain_c_rc + reverse_complement(toehold_b)
        full_linker = f"{linker_1}&{linker_2}"
        intended_structure = (
            "." * TOEHOLD_LENGTH
            + "(" * THIRD_DOMAIN_LENGTH
            + ")" * THIRD_DOMAIN_LENGTH
            + "." * TOEHOLD_LENGTH
        )

        md = RNA.md()
        md.temperature = float(temperature_C)
        md.salt = float(salt_M)
        md.dangles = 2
        # Require the intended, unslipped C-C* duplex to be the actual MFE.
        compound = RNA.fold_compound(full_linker, md)
        mfe_structure, mfe_energy = compound.mfe()
        intended_energy = float(compound.eval_structure(intended_structure))
        intended_is_mfe = abs(intended_energy - float(mfe_energy)) <= 1e-5
        slippage_free = mfe_structure == intended_structure
        if not intended_is_mfe or not slippage_free:
            continue

        return {
            "domain_C": domain_c,
            "domain_C_rc": domain_c_rc,
            "linker_strand_1": linker_1,
            "linker_strand_2": linker_2,
            "full_linker": full_linker,
            "linker_mfe_structure": mfe_structure,
            "linker_intended_structure": intended_structure,
            "linker_mfe_kcal_per_mol": float(mfe_energy),
            "linker_intended_energy_kcal_per_mol": intended_energy,
            "linker_intended_is_mfe": intended_is_mfe,
            "linker_slippage_free": slippage_free,
            "domain_C_self_delta_g_kcal_per_mol": self_energy,
            "domain_C_cross_delta_g_kcal_per_mol": cross_energies,
            "domain_C_evaluated_candidates": evaluated_count,
        }
    return None


def make_nanostar_pair_buckets(
    nanostar_count: int = NANOSTAR_COUNT,
) -> list[tuple[int, int]]:
    """Return every unordered, one-based pair of nanostar identifiers.

    With the default database size this produces ``C(16, 2) = 120`` buckets,
    beginning with ``(1, 2)`` and ending with ``(15, 16)``.
    """
    if nanostar_count < 2:
        raise ValueError("nanostar_count must be at least two.")
    return list(combinations(range(1, nanostar_count + 1), 2))


def screen_linker_toeholds(
    nanostar_pairs: Iterable[tuple[int, int]] | None = None,
    *,
    workbook: str | Path = DEFAULT_WORKBOOK,
    threshold: float = DEFAULT_THRESHOLD_KCAL_PER_MOL,
    binding_target: float = DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL,
    binding_tolerance: float = DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run the complete A/B/C/D linker design for each nanostar pair.

    This delegates screening to ``orthogonal.py``. Consequently, A, B, RC(A),
    and RC(B) must be unstructured; A and B must pass the homodimer criterion;
    and all unintended A/B orientation interactions must have Delta G greater
    than or equal to ``threshold``. Intended A-RC(A) and B-RC(B) binding is
    excluded from the cross-binding screen.

    After A/B selection, the function screens C and its two-strand linker, then
    D, both D-linkers, and both blockers. Finally it reconstructs the complete
    four strands of each selected nanostar with A assigned to the first and B
    to the second. One detailed dictionary is returned per input pair.

    ``complete`` means every structural, orthogonality, binding-window, and
    slippage condition in this file passed. Temperature-dependent occupancy is
    not screened here; it is calculated later by ``screening_pipeline.py``.
    Each run uses fresh randomness.
    """
    # No explicit pair list means all 120 combinations.
    pairs = list(
        make_nanostar_pair_buckets()
        if nanostar_pairs is None
        else nanostar_pairs
    )
    if binding_tolerance < 0:
        raise ValueError("binding_tolerance must be nonnegative.")
    results: list[dict[str, Any]] = []

    # Process buckets sequentially and retain failed buckets for diagnostics.
    for bucket_number, (nanostar_a, nanostar_b) in enumerate(pairs, start=1):
        if nanostar_a == nanostar_b:
            raise ValueError("A bucket must contain two different nanostars.")
        # Stage 1: choose orthogonal 8-mer A/B toeholds near the energy target.
        sequences, energies, evaluated_count, binding_energies = _screen_toehold_pair(
            threshold=threshold,
            binding_target=binding_target,
            binding_tolerance=binding_tolerance,
            temperature_C=temperature_C,
            salt_M=salt_M,
            max_candidates=max_candidates,
            workers=workers,
        )
        complete = len(sequences) == 2 and bool(binding_energies)
        toehold_a = sequences[0] if sequences else None
        toehold_b = sequences[1] if complete else None

        # Defensive confirmation that A, B, A*, and B* are all unstructured.
        linker_structures: dict[str, str] = {}
        secondary_structure_free = False
        if complete:
            linker_sequences = {
                "A": toehold_a,
                "RC(A)": reverse_complement(toehold_a),
                "B": toehold_b,
                "RC(B)": reverse_complement(toehold_b),
            }
            linker_structures = {
                name: fold(sequence, temperature_C, salt_M)[0]
                for name, sequence in linker_sequences.items()
            }
            secondary_structure_free = all(
                structure == "." * TOEHOLD_LENGTH
                for structure in linker_structures.values()
            )
            complete = complete and secondary_structure_free

        # Stage 2: choose C and construct the unslipped A*-C & C*-B* linker.
        third_domain_result = None
        if complete:
            third_domain_result = _screen_third_domain(
                toehold_a,
                toehold_b,
                threshold=threshold,
                temperature_C=temperature_C,
                salt_M=salt_M,
                max_candidates=max_candidates,
            )
            complete = third_domain_result is not None

        # Start the result record even if a later stage fails.
        result = {
            "bucket_number": bucket_number,
            "nanostar_A": nanostar_a,
            "nanostar_B": nanostar_b,
            "toehold_A": toehold_a,
            "toehold_A_rc": (
                reverse_complement(toehold_a) if toehold_a else None
            ),
            "toehold_B": toehold_b,
            "toehold_B_rc": (
                reverse_complement(toehold_b) if toehold_b else None
            ),
            "interaction_delta_g_kcal_per_mol": (
                energies.get(tuple(sorted((toehold_a, toehold_b))))
                if toehold_a and toehold_b
                else None
            ),
            "cognate_binding_delta_g_kcal_per_mol": binding_energies,
            "binding_target_kcal_per_mol": binding_target,
            "binding_tolerance_kcal_per_mol": binding_tolerance,
            "secondary_structures": linker_structures,
            "secondary_structure_free": secondary_structure_free,
            "evaluated_candidates": evaluated_count,
            "complete": complete,
        }
        if third_domain_result is not None:
            result.update(third_domain_result)
        else:
            result.update(
                {
                    "domain_C": None,
                    "domain_C_rc": None,
                    "linker_strand_1": None,
                    "linker_strand_2": None,
                    "full_linker": None,
                    "linker_intended_is_mfe": False,
                }
            )

        # Stage 3: choose D, build both D-linkers, and design the blockers.
        fourth_domain_result = None
        if complete:
            fourth_domain_result = _screen_domain_d_and_blockers(
                toehold_a,
                toehold_b,
                third_domain_result["domain_C"],
                threshold=threshold,
                temperature_C=temperature_C,
                salt_M=salt_M,
                max_candidates=max_candidates,
            )
            complete = fourth_domain_result is not None
            result["complete"] = complete
        if fourth_domain_result is not None:
            result.update(fourth_domain_result)
            result["nanostar_A_strands"] = _nanostar_strands_with_toehold(
                nanostar_a, toehold_a, workbook
            )
            result["nanostar_B_strands"] = _nanostar_strands_with_toehold(
                nanostar_b, toehold_b, workbook
            )
        else:
            result.update(
                {
                    "domain_D": None,
                    "linker_Bstar_D_Bstar": None,
                    "linker_Astar_D_Astar": None,
                    "blocker_1": None,
                    "blocker_2": None,
                    "nanostar_A_strands": None,
                    "nanostar_B_strands": None,
                }
            )
        # Print complete constructs for direct inspection and save the same
        # values in the result consumed by evaluation.py/the pipeline.
        results.append(result)
        print(
            f"Linker bucket {bucket_number}/{len(pairs)}: "
            f"NS pair (NS{nanostar_a}, NS{nanostar_b}), complete={complete}"
        )
        if complete:
            print(f"  Accepted bucket [A, B]: [{toehold_a}, {toehold_b}]")
            print(
                "  Domains: "
                f"A={toehold_a}, A*={result['toehold_A_rc']}; "
                f"B={toehold_b}, B*={result['toehold_B_rc']}; "
                f"C={result['domain_C']}, C*={result['domain_C_rc']}; "
                f"D={result['domain_D']}, D*={result['domain_D_rc']}"
            )
            print(
                "  Cognate binding Delta G: "
                f"A-A*={binding_energies['A-A*']:.3f}, "
                f"B-B*={binding_energies['B-B*']:.3f} kcal/mol"
            )
            print(f"  Nanostar A strands: {result['nanostar_A_strands']}")
            print(f"  Nanostar B strands: {result['nanostar_B_strands']}")
            print(f"  Linker 1 (A*-C & C*-B*): {result['full_linker']}")
            print(f"  Linker 2 (B*-D-B*): {result['linker_Bstar_D_Bstar']}")
            print(f"  Linker 3 (A*-D-A*): {result['linker_Astar_D_Astar']}")
            print(f"  Blockers: [{result['blocker_1']}, {result['blocker_2']}]")

    return results


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
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_C)
    parser.add_argument("--salt", type=float, default=DEFAULT_SALT_M)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    screened = screen_linker_toeholds(
        threshold=args.threshold,
        binding_target=args.binding_target,
        binding_tolerance=args.binding_tolerance,
        temperature_C=args.temperature,
        salt_M=args.salt,
        max_candidates=args.max_candidates,
        workers=args.workers,
    )
    completed = sum(result["complete"] for result in screened)
    print(f"Completed {completed}/{len(screened)} linker buckets.")

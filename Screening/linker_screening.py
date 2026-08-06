"""Design orthogonal linkers for two selected nanostars."""


from pathlib import Path
import random
import sys
from typing import Any, Sequence

import ViennaRNA as RNA

try:
    from .extract_ns_arms import DEFAULT_WORKBOOK, get_ns_arms
    from .evaluation import thermal_binding_metrics
except ImportError:
    from extract_ns_arms import DEFAULT_WORKBOOK, get_ns_arms
    from evaluation import thermal_binding_metrics


SURF_DIRECTORY = Path(__file__).resolve().parents[1] / "SURF2026_test"
if str(SURF_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SURF_DIRECTORY))

from orthogonal import (
    DEFAULT_SALT_M,
    DEFAULT_TEMPERATURE_C,
    DEFAULT_THRESHOLD_KCAL_PER_MOL,
    directional_binding_delta_g,
    fold,
    interaction_delta_g,
    next_unstructured_sequence,
    register_unstructured_sequence,
    reverse_complement,
    self_interaction_delta_g,
)


TOEHOLD_LENGTH = 8
THIRD_DOMAIN_LENGTH = 6
D_DOMAIN_LENGTH = 8
BLOCKER_A_OVERLAP_LENGTH = 3
DEFAULT_BLOCKER_HEAT_C = 35.0
BLOCKER_HOLD_TEMPERATURE_C = 25.0
BLOCKER_CONCENTRATION_M = 1e-6
DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL = -8.0
DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL = 0.5
TRIES_BEFORE_BACKTRACK = 1_000


def _nanostar_strands_with_toehold(
    arms: Sequence[str], toehold: str
) -> tuple[str, str, str, str]:

    """Function that takes in the number of nanostars you need, extracts nanostar arm from Yancheng's
    workbook, citation needed once publication is out (shhhhh). """

    if len(arms) != 4:
        raise ValueError("A nanostar must contain exactly four arms.")

    return tuple(
        reverse_complement(arms[index - 1])
        + "TT"
        + arms[index]
        + "T"
        + toehold
        for index in range(4)
    )

def md_env(temp, salt):
    """Avoid repetition with ViennaRNA.md() calls."""
    md = RNA.md()
    md.temperature = float(temp)
    md.salt = float(salt)
    md.dangles = 2
    return md

def _screen_domain_d_and_blockers(toehold_a: str, toehold_b: str, domain_c: str, *, threshold: float, temperature_C: float, salt_M: float, heat: float):
    """
        Part 1: Screen for a domain D that doesn't interact much with toeholds A, B, or C. 
        Part 2: Make sure blockers doesn't fold up on itself when paired with other domains (might be unnecessary).
    """

    rng = random.Random()
    seen: set[str] = set()
    monomer_mfes: dict[str, float] = {}
    reference_domains = (toehold_a, toehold_b, domain_c)

    # TODO: make sure that this isn't checked twice
    for domain in reference_domains:
        if not register_unstructured_sequence(
            domain, monomer_mfes, temperature_C, salt_M
        ):
            return None

    a_star = reverse_complement(toehold_a)
    b_star = reverse_complement(toehold_b)

    md = md_env(temperature_C, salt_M)

    evaluated_count = 0
    while evaluated_count < TRIES_BEFORE_BACKTRACK:
        evaluated_count += 1
        # TODO: I don't like throwing errors

        # finding strands first
        try:
            domain_d = next_unstructured_sequence(rng, D_DOMAIN_LENGTH, seen, monomer_mfes, temperature_C, salt_M, three_letter=True)
        except RuntimeError:
            return None

        self_energy = self_interaction_delta_g(domain_d, monomer_mfes, temperature_C, salt_M)
        cross_energies = {
            name: interaction_delta_g(domain_d, domain, monomer_mfes, temperature_C, salt_M)
            for name, domain in zip(("D-A", "D-B", "D-C"), reference_domains)
        }
        if min(self_energy, *cross_energies.values()) < threshold:
            continue

        # Condition 1: check if the whole linker is unstructured

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

        # TODO: Ask Yancheng/Erik about this design choice for even-length domain. Makes sense to me though. 
        if len(domain_d) % 2:
            raise ValueError("Domain D must have an even length.")
        half_d = len(domain_d) // 2

        # Each blocker covers half of D and the three adjacent bases of A*.
        # Those A* bases are complementary to the terminal three bases of the
        # corresponding A domain, so the blockers displace A as well as
        # occupying D instead of binding D alone.
        left_target = a_star[-BLOCKER_A_OVERLAP_LENGTH:] + domain_d[:half_d]
        right_target = domain_d[half_d:] + a_star[:BLOCKER_A_OVERLAP_LENGTH]
        blocker_1 = reverse_complement(left_target)
        blocker_2 = reverse_complement(right_target)
        if not all(
            register_unstructured_sequence(blocker, monomer_mfes, temperature_C, salt_M)
            for blocker in (blocker_1, blocker_2)
        ):
            continue

        blocker_pair_energy = interaction_delta_g(blocker_1, blocker_2, monomer_mfes, temperature_C, salt_M)
        if blocker_pair_energy < threshold:
            continue

        # B is present in the same system.  Reject blockers that can bind it,
        # rather than checking only blocker-blocker and blocker-linker binding.
        blocker_b_energies = {
            "blocker_1-B": interaction_delta_g(
                blocker_1, toehold_b, monomer_mfes, temperature_C, salt_M
            ),
            "blocker_2-B": interaction_delta_g(
                blocker_2, toehold_b, monomer_mfes, temperature_C, salt_M
            ),
        }
        if min(blocker_b_energies.values()) < threshold:
            continue

        # Fold each blocker against the entire A*-D-A* strand and require it
        # to bind exactly its assigned half of D.
        blocker_checks = {}
        for name, blocker, target_start in (
            (
                "blocker_1",
                blocker_1,
                len(a_star) - BLOCKER_A_OVERLAP_LENGTH,
            ),
            ("blocker_2", blocker_2, len(a_star) + half_d),
        ):
            target_length = len(blocker)
            intended = "." * target_start + "(" * target_length + "." * (len(linker_ada) - target_start - target_length) + ")" * target_length
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
        if not all(check["slippage_free"] for check in blocker_checks.values()):
            continue

        blocker_target_energies = {
            "blocker_1-target": directional_binding_delta_g(
                blocker_1, left_target, temperature_C, salt_M
            ),
            "blocker_2-target": directional_binding_delta_g(
                blocker_2, right_target, temperature_C, salt_M
            ),
        }

        blocker_thermal_metrics = {}
        thermal_pass = True
        for name, blocker, target in (
            ("blocker_1", blocker_1, left_target),
            ("blocker_2", blocker_2, right_target),
        ):
            metrics = thermal_binding_metrics(
                blocker,
                target,
                concentration_M=BLOCKER_CONCENTRATION_M,
                target_temperature_C=heat,
                fit_temperatures_C=(BLOCKER_HOLD_TEMPERATURE_C, heat),
                salt_M=salt_M,
            )
            at_hold = next(
                point for point in metrics["curve"]
                if point["temperature_C"] == BLOCKER_HOLD_TEMPERATURE_C
            )
            at_heat = next(
                point for point in metrics["curve"]
                if point["temperature_C"] == heat
            )
            metrics["bound_fraction_at_25C"] = at_hold["bound_fraction"]
            metrics["off_fraction_at_heat"] = at_heat["off_fraction"]
            metrics["passes_thermal_screen"] = (
                at_hold["bound_fraction"] >= 0.90
                and at_heat["off_fraction"] >= 0.50
            )
            blocker_thermal_metrics[name] = metrics
            thermal_pass = thermal_pass and metrics["passes_thermal_screen"]
        if not thermal_pass:
            continue

        return {
            "domain_D": domain_d,
            "domain_D_rc": reverse_complement(domain_d),
            "linker_Bstar_D_Bstar": linker_bdb,
            "linker_Astar_D_Astar": linker_ada,
            "D_linker_structures": linker_structures,
            "blocker_1": blocker_1,
            "blocker_2": blocker_2,
            "blocker_1_Astar_overlap": BLOCKER_A_OVERLAP_LENGTH,
            "blocker_2_Astar_overlap": BLOCKER_A_OVERLAP_LENGTH,
            "blocker_pair_delta_g_kcal_per_mol": blocker_pair_energy,
            "blocker_B_delta_g_kcal_per_mol": blocker_b_energies,
            "blocker_target_binding_kcal_per_mol": blocker_target_energies,
            "blocker_heat_C": heat,
            "blocker_hold_temperature_C": BLOCKER_HOLD_TEMPERATURE_C,
            "blocker_concentration_M": BLOCKER_CONCENTRATION_M,
            "blocker_thermal_metrics": blocker_thermal_metrics,
            "blockers_pass_thermal_screen": True,
            "blocker_slippage_checks": blocker_checks,
            "domain_D_self_delta_g_kcal_per_mol": self_energy,
            "domain_D_cross_delta_g_kcal_per_mol": cross_energies,
            "domain_D_evaluated_candidates": evaluated_count,
        }
    return None

def _screen_toehold_pair(
    *, threshold: float, binding_target: float, binding_tolerance: float, temperature_C: float, salt_M: float,
    ) -> tuple[list[str], dict[tuple[str, str], float], int, dict[str, float]]:
    """
        Find two orthogonal linker pairs and evaluate them. 
    """
    rng = random.Random()
    seen: set[str] = set()
    monomer_mfes: dict[str, float] = {}
    evaluated_total = 0
    while evaluated_total < TRIES_BEFORE_BACKTRACK:
        evaluated_total += 1
        try:
            sequences = [
                next_unstructured_sequence(
                    rng, TOEHOLD_LENGTH, seen, monomer_mfes,
                    temperature_C, salt_M, three_letter=True,
                )
                for _ in range(2)
            ]
        except RuntimeError:
            return [], {}, evaluated_total, {}

        self_energies = {
            (sequence, sequence): self_interaction_delta_g(
                sequence, monomer_mfes, temperature_C, salt_M
            )
            for sequence in sequences
        }
        pair_key = tuple(sorted(sequences))
        pair_energy = interaction_delta_g(
            sequences[0], sequences[1], monomer_mfes, temperature_C, salt_M
        )
        energies = {**self_energies, pair_key: pair_energy}
        if min(*self_energies.values(), pair_energy) < threshold:
            continue

        binding_energies = {
            "A-A*": directional_binding_delta_g(sequences[0], reverse_complement(sequences[0]), temperature_C, salt_M),
            "B-B*": directional_binding_delta_g(sequences[1], reverse_complement(sequences[1]), temperature_C, salt_M),
        }
        if all(
            abs(energy - binding_target) <= binding_tolerance
            for energy in binding_energies.values()
        ):
            return sequences, energies, evaluated_total, binding_energies

    return [], {}, evaluated_total, {}


def _screen_third_domain(
    toehold_a: str, toehold_b: str, *,
    threshold: float, temperature_C: float, salt_M: float) -> dict[str, Any] | None:
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

    # TODO: Not sure if this check is 100% needed. 
    for toehold in (toehold_a, toehold_b):
        if not register_unstructured_sequence(
            toehold, monomer_mfes, temperature_C, salt_M
        ):
            return None

    evaluated_count = 0
    while evaluated_count < TRIES_BEFORE_BACKTRACK:
        evaluated_count += 1
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


def screen_linker_toeholds(
    nanostar_a_arms: Sequence[str],
    nanostar_b_arms: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD_KCAL_PER_MOL,
    binding_target: float = DEFAULT_TOEHOLD_BINDING_TARGET_KCAL_PER_MOL,
    binding_tolerance: float = DEFAULT_TOEHOLD_BINDING_TOLERANCE_KCAL_PER_MOL,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
    heat: float = DEFAULT_BLOCKER_HEAT_C,
) -> dict[str, Any]:
    """Design one complete linker system from two four-arm nanostars."""
    if len(nanostar_a_arms) != 4 or len(nanostar_b_arms) != 4:
        raise ValueError("Each nanostar must contain exactly four arms.")
    if binding_tolerance < 0:
        raise ValueError("binding_tolerance must be nonnegative.")
    if heat <= BLOCKER_HOLD_TEMPERATURE_C:
        raise ValueError(
            f"heat must be greater than {BLOCKER_HOLD_TEMPERATURE_C} deg C."
        )

    restart_count = 0
    while True:
        sequences, energies, evaluated_count, binding_energies = (
            _screen_toehold_pair(
                threshold=threshold,
                binding_target=binding_target,
                binding_tolerance=binding_tolerance,
                temperature_C=temperature_C,
                salt_M=salt_M,
            )
        )
        if len(sequences) != 2 or not binding_energies:
            restart_count += 1
            continue

        toehold_a, toehold_b = sequences
        third_domain_result = _screen_third_domain(
            toehold_a,
            toehold_b,
            threshold=threshold,
            temperature_C=temperature_C,
            salt_M=salt_M,
        )
        if third_domain_result is None:
            restart_count += 1
            continue

        fourth_domain_result = _screen_domain_d_and_blockers(
            toehold_a,
            toehold_b,
            third_domain_result["domain_C"],
            threshold=threshold,
            temperature_C=temperature_C,
            salt_M=salt_M,
            heat=heat,
        )
        if fourth_domain_result is None:
            restart_count += 1
            continue

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
        result = {
            "nanostar_A_arms": tuple(nanostar_a_arms),
            "nanostar_B_arms": tuple(nanostar_b_arms),
            "toehold_A": toehold_a,
            "toehold_A_rc": reverse_complement(toehold_a),
            "toehold_B": toehold_b,
            "toehold_B_rc": reverse_complement(toehold_b),
            "interaction_delta_g_kcal_per_mol": energies[
                tuple(sorted((toehold_a, toehold_b)))
            ],
            "cognate_binding_delta_g_kcal_per_mol": binding_energies,
            "binding_target_kcal_per_mol": binding_target,
            "binding_tolerance_kcal_per_mol": binding_tolerance,
            "secondary_structures": linker_structures,
            "secondary_structure_free": True,
            "evaluated_candidates": evaluated_count,
            "restart_count": restart_count,
            "complete": True,
            **third_domain_result,
            **fourth_domain_result,
            "nanostar_A_strands": _nanostar_strands_with_toehold(
                nanostar_a_arms, toehold_a
            ),
            "nanostar_B_strands": _nanostar_strands_with_toehold(
                nanostar_b_arms, toehold_b
            ),
        }
        print(
            f"Nanostar linker completed after {restart_count} backtracks."
        )
        print(f"  A={toehold_a}, B={toehold_b}")
        print(
            f"  C={result['domain_C']}, D={result['domain_D']}, "
            f"blockers=[{result['blocker_1']}, {result['blocker_2']}]"
        )
        return result
if __name__ == "__main__":
    # Default arm sequences copied from NS1 and NS2 in the workbook.
    nanostar_A_arms = (
        "GCGTCCGACACTGAACTATG",
        "TGTCAGCCGTGCTATCAAGA",
        "ACGGTCTGACCCGAAATAGT",
        "ATGTGGCCTACAGTGAATCC",
    )
    nanostar_B_arms = (
        "TTGACCACCTAGGATGCGTT",
        "CGGAGACTAGATGATTTCCG",
        "GATGTCTAACGATTCAGGCC",
        "CAAGTATCGGTGCTGATCCA",
    )

    # To choose different nanostars from the XLSX instead:
    # workbook_arms = get_ns_arms(DEFAULT_WORKBOOK)
    # nanostar_A_arms = workbook_arms[0:4]
    # nanostar_B_arms = workbook_arms[4:8]

    result = screen_linker_toeholds(nanostar_A_arms, nanostar_B_arms)
    print(f"Complete: {result['complete']}")

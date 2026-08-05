"""Build and verify four-arm nanostars from the screening workbook."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import ViennaRNA as RNA

try:
    from .extract_ns_arms import DEFAULT_WORKBOOK, SEQUENCE_COLUMN, get_ns_arms
except ImportError:  # Allow ``python Screening/verification.py``.
    from extract_ns_arms import DEFAULT_WORKBOOK, SEQUENCE_COLUMN, get_ns_arms

# Reuse orthogonal.py's ViennaRNA MFE helper. This does not run its
# orthogonality screening.
SURF_DIRECTORY = Path(__file__).resolve().parents[1] / "SURF2026_test"
if str(SURF_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SURF_DIRECTORY))

from orthogonal import fold  # noqa: E402

DEFAULT_TEMPERATURE_C = 20.0
DEFAULT_SALT_M = 1.0


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(str.maketrans("ATCG", "TAGC"))[::-1]


def get_spacers(workbook: str | Path = DEFAULT_WORKBOOK) -> list[str]:
    """Return the 16 unique nine-base spacers in workbook order."""
    excel_file = pd.ExcelFile(Path(workbook), engine="xlrd")
    sequences = None
    for sheet_name in excel_file.sheet_names:
        candidate = pd.read_excel(excel_file, sheet_name=sheet_name)
        if SEQUENCE_COLUMN in candidate.columns:
            sequences = candidate[SEQUENCE_COLUMN]
            break
    if sequences is None:
        raise ValueError(f"Could not find column {SEQUENCE_COLUMN!r}.")

    spacers: list[str] = []
    for value in sequences:
        if pd.isna(value) or not str(value).strip():
            continue
        spacer = "".join(str(value).split())[-9:]
        if spacer not in spacers:
            spacers.append(spacer)
    if len(spacers) != 16:
        raise ValueError(f"Expected 16 unique spacers, found {len(spacers)}.")
    return spacers


def build_nanostars(
    spacer_number: int = 1,
    workbook: str | Path = DEFAULT_WORKBOOK,
) -> list[tuple[str, str, str, str]]:
    """Construct every four-strand nanostar using spacer 1 through 16."""
    arms = get_ns_arms(workbook)
    if len(arms) % 4:
        raise ValueError(f"The arm count ({len(arms)}) is not divisible by four.")

    spacers = get_spacers(workbook)
    if not 1 <= spacer_number <= len(spacers):
        raise ValueError(f"spacer_number must be between 1 and {len(spacers)}.")
    spacer = spacers[spacer_number - 1]

    nanostars = []
    for offset in range(0, len(arms), 4):
        group = arms[offset : offset + 4]
        strands = tuple(
            reverse_complement(group[index - 1])
            + "TT"
            + group[index]
            + "T"
            + spacer
            for index in range(4)
        )
        nanostars.append(strands)
    return nanostars


def _ideal_nanostar_structure(arm_length: int, spacer_length: int) -> str:
    """Create the cyclic target structure used by NSDesign.py."""
    tail = "." * (1 + spacer_length)
    parts = [
        "(" * arm_length + ".." + "(" * arm_length + tail,
        ")" * arm_length + ".." + "(" * arm_length + tail,
        ")" * arm_length + ".." + "(" * arm_length + tail,
        ")" * arm_length + ".." + ")" * arm_length + tail,
    ]
    return "".join(parts)


def linker_binding_free_energy(
    nanostar_1: tuple[str, str, str, str] | list[str],
    nanostar_2: tuple[str, str, str, str] | list[str],
    linker_strands: str | list[str] | tuple[str, ...],
    linker_mfe_kcal_per_mol: float | None = None,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> float:
    """Return linker-binding Delta G for two four-strand nanostars.

    Each nanostar strand must include its toehold. ``linker_strands`` may be a
    single sequence or multiple strands. Following NSDesign.py, the bound MFE
    is compared with the two isolated nanostar MFEs and the isolated linker
    MFE. Following orthogonal.py, all MFEs use the same temperature, salt, and
    dangle settings.

    A negative result means that linker binding is thermodynamically favorable.
    If ``linker_mfe_kcal_per_mol`` is omitted, the linker is folded by itself
    (or as an isolated multistrand complex) to determine that value.
    """
    if len(nanostar_1) != 4 or len(nanostar_2) != 4:
        raise ValueError("Each nanostar must contain exactly four strands.")

    if isinstance(linker_strands, str):
        linkers = [linker_strands]
    else:
        linkers = list(linker_strands)
    if not linkers or any(not sequence for sequence in linkers):
        raise ValueError("At least one nonempty linker strand is required.")

    nanostar_1_sequence = "&".join(nanostar_1)
    nanostar_2_sequence = "&".join(nanostar_2)
    linker_sequence = "&".join(linkers)
    linked_sequence = "&".join([*nanostar_1, *nanostar_2, *linkers])

    nanostar_1_mfe = fold(nanostar_1_sequence, temperature_C, salt_M)[1]
    nanostar_2_mfe = fold(nanostar_2_sequence, temperature_C, salt_M)[1]
    linked_mfe = fold(linked_sequence, temperature_C, salt_M)[1]
    if linker_mfe_kcal_per_mol is None:
        linker_mfe_kcal_per_mol = fold(
            linker_sequence, temperature_C, salt_M
        )[1]

    return float(
        linked_mfe
        - nanostar_1_mfe
        - nanostar_2_mfe
        - linker_mfe_kcal_per_mol
    )


def verify_nanostars(
    spacer_number: int = 1,
    workbook: str | Path = DEFAULT_WORKBOOK,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
    salt_M: float = DEFAULT_SALT_M,
) -> list[dict[str, Any]]:
    """Return target-structure formation results for all nanostars.

    Formation follows NSDesign.py: the target cyclic structure must have the
    same energy as the MFE structure.
    """
    RNA.params_load_DNA_Mathews2004()
    arms = get_ns_arms(workbook)
    nanostars = build_nanostars(spacer_number, workbook)
    spacer = get_spacers(workbook)[spacer_number - 1]

    md = RNA.md()
    md.temperature = float(temperature_C)
    md.salt = float(salt_M)
    md.dangles = 2

    reports: list[dict[str, Any]] = []
    for nanostar_index, strands in enumerate(nanostars, start=1):
        group = arms[(nanostar_index - 1) * 4 : nanostar_index * 4]
        ideal_structure = _ideal_nanostar_structure(len(group[0]), len(spacer))
        complex_sequence = "&".join(strands)
        compound = RNA.fold_compound(complex_sequence, md)
        mfe_structure, mfe_energy = compound.mfe()
        ideal_energy = float(compound.eval_structure(ideal_structure))
        ideal_is_mfe = bool(np.isclose(ideal_energy, mfe_energy))

        reports.append(
            {
                "nanostar_number": nanostar_index,
                "spacer_number": spacer_number,
                "spacer": spacer,
                "arms": tuple(group),
                "strands": strands,
                "mfe_structure": mfe_structure,
                "ideal_structure": ideal_structure,
                "mfe_energy_kcal_per_mol": float(mfe_energy),
                "ideal_energy_kcal_per_mol": ideal_energy,
                "ideal_is_mfe": ideal_is_mfe,
                "valid": ideal_is_mfe,
            }
        )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spacer",
        type=int,
        choices=range(1, 17),
        default=1,
        help="Spacer number to use for all nanostars (default: 1).",
    )
    args = parser.parse_args()
    results = verify_nanostars(spacer_number=args.spacer)
    for result in results:
        print(
            f"Nanostar {result['nanostar_number']:2d}: "
            f"valid={result['valid']}, ideal_MFE={result['ideal_is_mfe']}"
        )

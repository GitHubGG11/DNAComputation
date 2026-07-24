import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import ViennaRNA as RNA

from delta_g_estimator import reverse_complement_dna, setup_django


arm1 = "GCGTCCGACACTGAAC"
arm2 = "TGTCAGCCGTGCTATC"
arm3 = "ACGGTCTGACCCGAAA"
arm4 = "ATGTGGCCTACAGTGA"
N = 1000

RNA.params_load_DNA_Mathews2004()


def random_nanostar_curve(
    domainA="ACGTTGCAAGTC",
    linker="GGTA",
    middle="GTACT",
    G="CG",
    temperatures_C=range(20, 81, 5),
):
    """Generate a random nanostar design and fit Delta G = H - T*S.

    Explicit sequence arguments can be supplied to reproduce or evaluate a
    particular design. Energies, H, and S are all expressed in kcal/mol (with
    S in kcal/mol/K). The returned row can be passed to
    ``append_nanostar_rows(row, temperature_curve=curve)``.
    """
    bases = "ATCG"
    domainA = "".join(random.choice(bases) for _ in range(12))
    linker = "".join(random.choice(bases) for _ in range(4))
    middle = "".join(random.choice(bases) for _ in range(5))

    domainAc = reverse_complement_dna(domainA)
    linkerc = reverse_complement_dna(linker)
    middlec = reverse_complement_dna(middle)
    arm1c = reverse_complement_dna(arm1)
    arm2c = reverse_complement_dna(arm2)
    arm3c = reverse_complement_dna(arm3)
    arm4c = reverse_complement_dna(arm4)

    strands = [
        domainA + arm4c + "TT" + arm1 + G + domainAc + "T" + linker,
        domainA + arm1c + "TT" + arm2 + G + domainAc + "T" + linker,
        domainA + arm2c + "TT" + arm3 + G + domainAc + "T" + linker,
        domainA + arm3c + "TT" + arm4 + G + domainAc + "T" + linker,
    ]
    strand_lock = middle + "T" + linkerc + domainA + reverse_complement_dna(G)
    strand_lock_c = middlec + "T" + linkerc + domainA + reverse_complement_dna(G)
    linked_sequence = "&".join(
        strands + [strand_lock, strand_lock_c] + strands
    )
    unlinked_sequence = "&".join(strands + strands)

    arm_length = len(arm1)
    domain_length = len(domainA)
    bulge_length = len(G)
    toehold_length = len(linker)
    ideal_part = "".join(
        [
            "(" * domain_length + "(" * arm_length + ".." + "(" * arm_length
            + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
            ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length
            + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
            ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length
            + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
            ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length
            + "." * bulge_length + ")" * domain_length + "." * (toehold_length + 1),
        ]
    )

    curve = []
    for temperature_C in temperatures_C:
        md = RNA.md()
        md.temperature = float(temperature_C)
        md.salt = 1.0
        md.dangles = 2

        linked = RNA.fold_compound(linked_sequence, md)
        structure, linked_mfe = linked.mfe()
        unlinked = RNA.fold_compound(unlinked_sequence, md)
        unlinked_energy = unlinked.eval_structure(ideal_part + ideal_part)
        curve.append(
            {
                "temperature_C": float(temperature_C),
                "structure": structure,
                "mfe_kcal_per_mol": float(linked_mfe - unlinked_energy),
            }
        )

    if len(curve) < 2:
        raise ValueError("at least two temperatures are required for regression")
    temperatures_K = np.array(
        [point["temperature_C"] + 273.15 for point in curve], dtype=float
    )
    energies = np.array(
        [point["mfe_kcal_per_mol"] for point in curve], dtype=float
    )
    slope, intercept = np.polyfit(temperatures_K, energies, 1)

    row = {
        "arm1": arm1,
        "arm2": arm2,
        "arm3": arm3,
        "arm4": arm4,
        "middle": middle,
        "linker": linker,
        "A_Domain": domainA,
        "H": float(intercept),
        "S": float(-slope),
    }
    return row, curve


if __name__ == "__main__":
    nanostars = []
    with ProcessPoolExecutor() as executor:
        generated = executor.map(
            random_nanostar_curve,
            repeat(None, N),
            repeat(None, N),
            repeat(None, N),
            chunksize=1,
        )
        for iteration, (nanostar, melting_curve) in enumerate(generated, 1):
            nanostar["curve"] = melting_curve
            nanostars.append(nanostar)
            print(f"\rGenerated {iteration}/{N}", end="", flush=True)

    setup_django()
    from SURF2026_test.nanostar_database import append_nanostar_rows, wipe_nanostar_tables

    wipe_nanostar_tables()
    inserted_ids = append_nanostar_rows(nanostars)
    print(f"\nUploaded {len(inserted_ids)} nanostars")

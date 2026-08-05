import random
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import ViennaRNA as RNA

from delta_g_estimator import reverse_complement_dna, setup_django


arm1 = "GCGTCCGACACTGAAC"
arm2 = "TGTCAGCCGTGCTATC"
arm3 = "ACGGTCTGACCCGAAA"
arm4 = "ATGTGGCCTACAGTGA"
N = 50000
SAVE_BATCH_SIZE = 25
R =  0.001987

RNA.params_load_DNA_Mathews2004()


def random_nanostar_curve(
    domainA="ACGTTGCAAGTC",
    linker="GGTA",
    middle="GTACT",
    G="CT",
    temperatures_C=range(20, 40, 10),
):
    """Generate a random nanostar design and fit Delta G = H - T*S.

    Explicit sequence arguments can be supplied to reproduce or evaluate a
    particular design. Energies, H, and S are all expressed in kcal/mol (with
    S in kcal/mol/K). The returned row can be passed to
    ``append_nanostar_rows(row, temperature_curve=curve)``.
    """

    bases = "ATC"
    domainA = "".join(random.choice(bases) for _ in range(12))
    linker = "".join(random.choice(bases) for _ in range(8))
    middle = "".join(random.choice(bases + "G") for _ in range(15))


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
        strands + [strand_lock] + strands + [strand_lock_c]
    )
    unlinked_sequence = "&".join(strands + strands)

    arm_length = len(arm1)
    domain_length = len(domainA)
    bulge_length = len(G)
    toehold_length = len(linker)

    middle_length = len(middle)
    linker_end_length = len(linkerc) + len(domainA) + len(G)

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

    ns_left_parts = [
        "." * domain_length + "(" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length + "(" * (bulge_length + domain_length) + "." + "(" * toehold_length,
    ]

    ns_right_parts = [
        "." * domain_length + "(" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length + ")" * (bulge_length + domain_length) + "." + ")" * toehold_length,
    ]

    # evaluate kinetic rates @ 20 degerees

    step1 = [
        "(" * domain_length + "(" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length + "." * bulge_length + ")" * domain_length + "." * (toehold_length + 1),
        "." * len(strand_lock),
    ]
    step1 = "".join(step1)

    step2 = [
        "(" * domain_length + "(" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length + "." * bulge_length + ")" * domain_length + "." + "(" * toehold_length,
        "." * (middle_length + 1) + ")" * toehold_length + "." * (domain_length + bulge_length),
    ]
    step2 = "".join(step2)

    step3 = [
        "." * domain_length + "(" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length + "." * bulge_length + "(" * domain_length + "." + "(" * toehold_length,
        "." * (middle_length + 1) + ")" * (toehold_length + domain_length) + "." * bulge_length,
    ]
    step3 = "".join(step3)

    step4 = [
        "." * domain_length + "(" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + "(" * arm_length + "." * bulge_length + "(" * domain_length + "." * (toehold_length + 1),
        ")" * domain_length + ")" * arm_length + ".." + ")" * arm_length + "(" * (bulge_length + domain_length) + "." + "(" * toehold_length,
        "." * (middle_length + 1) + ")" * (toehold_length + domain_length + bulge_length),
    ]

    step4 = "".join(step4)

    step_test = "&".join(
        strands + [strand_lock]
    )

    md = RNA.md()
    md.temperature = 20.0
    md.salt = 1.0
    md.dangles = 2

    step1G = RNA.fold_compound(step_test, md).eval_structure(step1)
    step2G = RNA.fold_compound(step_test, md).eval_structure(step2)
    step3G = RNA.fold_compound(step_test, md).eval_structure(step3)
    step4G = RNA.fold_compound(step_test, md).eval_structure(step4)
    k1 = 1e6
    k2 = 1
    k3m = 0.1
    k1m = k1*np.exp((step2G - step1G) / (R * (20.0 + 273.15)))
    k2m = k2*np.exp((step3G - step2G) / (R * (20.0 + 273.15)))
    k3 = k3m*np.exp((step3G - step4G) / (R * (20.0 + 273.15)))

    kmeff = (k1m * k2m * k3m) / (k1m * k2m + k1m * k3m + k2m * k3m + k1m * k3 + k3m * k2 + k2* k3)
    keff = kmeff * np.exp((step1G - step4G) / (R * (20.0 + 273.15)))

    # print(step1G, step2G, step3G, step4G)
    # print(k1, k1m, k2, k2m, k3, k3m)
    # print(kmeff, keff)
    # evaluate ideal structure

    strand_lock_part = "(" * middle_length + "." + "(" * linker_end_length
    strand_lock_c_part = ")" * middle_length + "." + ")" * linker_end_length

    ideal_parts = ns_left_parts + [strand_lock_part] + ns_right_parts + [strand_lock_c_part]
    ideal_structure = "".join(ideal_parts)


    curve = []
    for temperature_C in temperatures_C:
        md = RNA.md()
        md.temperature = float(temperature_C)
        md.salt = 1.0
        md.dangles = 2

        linked = RNA.fold_compound(linked_sequence, md)
        structure, linked_mfe = linked.mfe()

        linked_ideal_mfe = linked.eval_structure(ideal_structure)
        ideal_mfe_difference = linked_ideal_mfe - linked_mfe
        if temperature_C < 25 and not np.isclose(ideal_mfe_difference, 0.0):
            return None, None
        unlinked = RNA.fold_compound(unlinked_sequence, md)
        unlinked_energy = unlinked.eval_structure(ideal_part + ideal_part)

        middle_link = RNA.fold_compound("&".join([strand_lock, strand_lock_c]), md)
        mid_structure, mid_mfe = middle_link.mfe()

        n_m = len(strand_lock) - middle_length
        ideal_mid_structure = "(" * middle_length + "." * n_m + ")" * middle_length + "." * n_m

        if temperature_C < 25 and not mid_structure == ideal_mid_structure:
            return None, None

        curve.append(
            {
                "temperature_C": float(temperature_C),
                "structure": structure,
                "mfe_kcal_per_mol": float(linked_mfe - unlinked_energy - mid_mfe),
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
        "kmeff": float(np.log10(kmeff)),
        "keff": float(np.log10(keff)),
        "k1": float(k1),
        "k2": float(k2),
        "k3": float(k3),
        "k1m": float(k1m),
        "k2m": float(k2m),
        "k3m": float(k3m),
        "full_arm1": strands[0],
        "full_arm2": strands[1],
        "full_arm3": strands[2],
        "full_arm4": strands[3],
        "upper_linker": strand_lock,
        "lower_linker": strand_lock_c,
    }
    return row, curve


if __name__ == "__main__":
    setup_django()
    from SURF2026_test.nanostar_database import append_nanostar_rows, wipe_nanostar_tables

    wipe_nanostar_tables()

    pending_rows = []
    accepted_count = 0
    uploaded_count = 0
    completed_count = 0
    worker_count = os.cpu_count() or 1
    max_in_flight = worker_count * 4

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        submitted_count = min(N, max_in_flight)
        pending_futures = {
            executor.submit(random_nanostar_curve)
            for _ in range(submitted_count)
        }

        while pending_futures:
            completed_futures, pending_futures = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in completed_futures:
                nanostar, melting_curve = future.result()
                completed_count += 1

                if submitted_count < N:
                    pending_futures.add(executor.submit(random_nanostar_curve))
                    submitted_count += 1

                if nanostar is not None and melting_curve is not None:
                    nanostar["curve"] = melting_curve
                    pending_rows.append(nanostar)
                    accepted_count += 1

                    if len(pending_rows) >= SAVE_BATCH_SIZE:
                        uploaded_count += len(append_nanostar_rows(pending_rows))
                        pending_rows.clear()

                print(
                    f"\rGenerated {completed_count}/{N}; "
                    f"accepted {accepted_count}; saved {uploaded_count}",
                    end="",
                    flush=True,
                )

    if pending_rows:
        uploaded_count += len(append_nanostar_rows(pending_rows))

    print(
        f"\nAccepted {accepted_count} sequences; "
        f"uploaded {uploaded_count} nanostars"
    )

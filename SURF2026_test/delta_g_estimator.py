import numpy as np
import ViennaRNA as RNA
import matplotlib.pyplot as plt
import os
import random
import sys
from pathlib import Path


R_KCAL_PER_MOL_K = 0.0019872041
R_KJ_PER_MOL_K = 0.008314462618
KCAL_TO_KJ = 4.184

RNA.params_load_DNA_Mathews2004() 

RNA.cvar.salt=1
RNA.cvar.dangles=2

middle_length = 6
linker_length = 4

BASES = "ATCG"
LINKER_STRUCTURE = "(" * linker_length + ")" * linker_length
MIDDLE_STRUCTURE = "(" * middle_length + ")" * middle_length
DEFAULT_TARGET_STRUCTURE =  LINKER_STRUCTURE + "." + MIDDLE_STRUCTURE + "." + LINKER_STRUCTURE


def reverse_complement_dna(sequence):
    bp = str.maketrans("ATCGatcg", "TAGCtagc")
    return sequence.translate(bp)[::-1].upper()


def random_dna(length, rng):
    return "".join(rng.choice(BASES) for _ in range(length))


def linker_complex_sequence(middle_strand, linker):
    middle_strand = middle_strand.upper()
    linker = linker.upper()
    middle_strand_rc = reverse_complement_dna(middle_strand)
    return linker + "T" + middle_strand + '&' + middle_strand_rc + "T" +linker[::-1]


def linker_addition_compound_sequence(middle_strand, linker):
    linker = linker.upper()
    linker_rc = reverse_complement_dna(linker)
    return (
        linker_rc
        + '&'
        + linker_complex_sequence(middle_strand, linker)
        + '&'
        + linker_rc[::-1]
    )


def expected_target_length(middle_length, linker_length):
    return 2 * middle_length + 4 * linker_length + 2


def mfe_structure(sequence, temperature=20.0):
    RNA.cvar.temperature = float(temperature)
    md = RNA.md()
    compound = RNA.fold_compound(sequence, md)
    return compound.mfe()


def valid_mfe_results(
    middle_strand,
    linker,
    target_structure=DEFAULT_TARGET_STRUCTURE,
    validation_temperatures=(20.0,),
):
    compound_sequence = linker_addition_compound_sequence(middle_strand, linker)
    results = []

    for temperature in validation_temperatures:
        structure, energy = mfe_structure(compound_sequence, temperature)
        if structure != target_structure:
            return None
        results.append({
            "temperature_C": float(temperature),
            "structure": structure,
            "mfe_kcal_per_mol": float(energy),
        })

    return compound_sequence, results


def free_energy_curve(middle_strand, linker, temperatures):
    compound_sequence = linker_addition_compound_sequence(middle_strand, linker)
    curve = []

    for temperature in temperatures:
        structure, energy = mfe_structure(compound_sequence, temperature)
        curve.append({
            "temperature_C": float(temperature),
            "structure": structure,
            "mfe_kcal_per_mol": float(energy),
        })

    return compound_sequence, curve


def fit_free_energy_thermodynamics(curve):
    temperatures_K = np.array(
        [point["temperature_C"] + 273.15 for point in curve],
        dtype=float,
    )
    energies = np.array([point["mfe_kcal_per_mol"] for point in curve], dtype=float)

    if len(temperatures_K) < 2:
        return float(energies[0]), 0.0

    slope, intercept = np.polyfit(temperatures_K, energies, 1)
    H = float(intercept)
    S = float(-slope)
    return H, S


def update_progress_bar(checked, valid_count, total_trials=None, width=30):
    if total_trials:
        fraction = min(checked / total_trials, 1.0)
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r[{bar}] {checked}/{total_trials} "
            f"valid={valid_count}"
        )
    else:
        message = f"\rchecked={checked} valid={valid_count}"

    sys.stdout.write(message)
    sys.stdout.flush()


def setup_django():
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.append(str(base_dir))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SURF2026.settings")
    import django

    django.setup()


def replace_valid_linker_model_rows():
    setup_django()
    from django.db import transaction
    from SURF2026_test.models import LinkerTemperatureCurve, ValidLinkerSequence

    with transaction.atomic():
        ValidLinkerSequence.objects.all().delete()
        LinkerTemperatureCurve.objects.all().delete()

    return LinkerTemperatureCurve, ValidLinkerSequence


def insert_valid_linker_model_row(
    LinkerTemperatureCurve,
    ValidLinkerSequence,
    middle,
    linker,
    compound,
    curve,
    target_structure,
    validation_temperatures,
):
    H, S = fit_free_energy_thermodynamics(curve)
    curve_row = LinkerTemperatureCurve.objects.create(
        compound=compound,
        target_structure=target_structure,
        validation_temperatures_C=[float(T) for T in validation_temperatures],
        curve=curve,
    )
    ValidLinkerSequence.objects.create(
        middle=middle,
        linker=linker,
        middle_rc=reverse_complement_dna(middle),
        linker_rc=reverse_complement_dna(linker),
        H=H,
        S=S,
        curve=curve_row,
    )
    return H, S, curve_row.id


def replace_valid_linker_django_tables(
    middle_length,
    linker_length,
    target_structure=DEFAULT_TARGET_STRUCTURE,
    validation_temperatures=(20.0,),
    curve_start=20.0,
    curve_end=80.0,
    curve_step=1.0,
    max_valid=100,
    max_trials=100000,
    seed=None,
    print_every=10000,
):
    """
    Replace Django/Postgres tables with randomly sampled exact-MFE linker matches.

    Tables:

        linker_temperature_curves:
            one row per valid sequence, with the full temperature curve as JSON

        valid_linker_sequences:
            one row per valid sequence, referencing linker_temperature_curves.id
    """
    target_length = len(target_structure)
    compound_length = expected_target_length(middle_length, linker_length)
    if target_length != compound_length:
        raise ValueError(
            "target_structure length does not match compound length: "
            f"{target_length} != {compound_length}. "
            "For this format, length = 2*middle_length + 4*linker_length."
        )

    curve_temperatures = np.arange(curve_start, curve_end + curve_step, curve_step)
    checked = 0
    valid_count = 0
    rng = random.Random(seed)
    LinkerTemperatureCurve, ValidLinkerSequence = replace_valid_linker_model_rows()
    total_trials = None if max_trials is None else max_trials

    while max_trials is None or checked < max_trials:
        checked += 1
        middle = random_dna(middle_length, rng)
        linker = random_dna(linker_length, rng)

        valid = valid_mfe_results(
            middle,
            linker,
            target_structure=target_structure,
            validation_temperatures=validation_temperatures,
        )
        if valid is not None:
            compound, curve = free_energy_curve(middle, linker, curve_temperatures)
            insert_valid_linker_model_row(
                LinkerTemperatureCurve=LinkerTemperatureCurve,
                ValidLinkerSequence=ValidLinkerSequence,
                middle=middle,
                linker=linker,
                compound=compound,
                curve=curve,
                target_structure=target_structure,
                validation_temperatures=validation_temperatures,
            )
            valid_count += 1

        if checked == 1 or checked % 100 == 0 or (
            max_valid is not None and valid_count >= max_valid
        ):
            update_progress_bar(checked, valid_count, total_trials)

        if max_valid is not None and valid_count >= max_valid:
            break

    sys.stdout.write("\n")

    return {
        "checked": checked,
        "valid": valid_count,
    }


def print_valid_linker_free_energy_curves(
    middle_length,
    linker_length,
    target_structure=DEFAULT_TARGET_STRUCTURE,
    validation_temperatures=(20.0,),
    curve_start=20.0,
    curve_end=80.0,
    curve_step=1.0,
    max_valid=5,
    max_trials=100000,
    seed=None,
    print_every=10000,
):
    """
    Randomly sample middle/linker sequences and print exact MFE matches.

    The only iterated sequence variables are:

        middle
        linker

    Everything else is generated:

        middle_rc = reverse_complement_dna(middle)
        linker_rc = reverse_complement_dna(linker)

    The scored compound is:

        linker_rc & linker+middle & middle_rc+linker[::-1] & linker_rc[::-1]

    A sequence is printed only when its MFE structure exactly equals
    target_structure at every validation temperature. No mismatches are accepted.
    """
    target_length = len(target_structure)
    compound_length = expected_target_length(middle_length, linker_length)
    if target_length != compound_length:
        raise ValueError(
            "target_structure length does not match compound length: "
            f"{target_length} != {compound_length}. "
            "For this format, length = 2*middle_length + 4*linker_length."
        )

    curve_temperatures = np.arange(curve_start, curve_end + curve_step, curve_step)
    checked = 0
    valid_count = 0
    rng = random.Random(seed)
    total_trials = None if max_trials is None else max_trials

    while max_trials is None or checked < max_trials:
        checked += 1
        middle = random_dna(middle_length, rng)
        linker = random_dna(linker_length, rng)

        valid = valid_mfe_results(
            middle,
            linker,
            target_structure=target_structure,
            validation_temperatures=validation_temperatures,
        )
        if valid is not None:
            _, curve = free_energy_curve(middle, linker, curve_temperatures)
            H, S = fit_free_energy_thermodynamics(curve)
            valid_count += 1

        if checked == 1 or checked % 100 == 0 or (
            max_valid is not None and valid_count >= max_valid
        ):
            update_progress_bar(checked, valid_count, total_trials)

        if max_valid is not None and valid_count >= max_valid:
            sys.stdout.write("\n")
            return {
                "checked": checked,
                "valid": valid_count,
            }

    sys.stdout.write("\n")

    return {
        "checked": checked,
        "valid": valid_count,
    }


def melting_energies(start=20, end=80, step=1, middle_strand='CACCAC', linker='AAG'):
    temperatures = np.arange(start, end + step, step)
    energies = []

    compound_sequence = linker_addition_compound_sequence(middle_strand, linker)

    for temperature in temperatures:
        structure, energy = mfe_structure(compound_sequence, temperature)
        print(structure)
        energies.append(energy)

    return temperatures, energies


if __name__ == "__main__":
    replace_valid_linker_django_tables(
        middle_length=middle_length,
        linker_length=linker_length,
        validation_temperatures=(20.0,),
        curve_start=20.0,
        curve_end=80.0,
        curve_step=1.0,
        max_valid=1000,
        max_trials=100000,
        seed=1,
    )

"""Grow buckets of mutually orthogonal random DNA strands.

Each candidate and its reverse complement must be unstructured by themselves.
A candidate joins a bucket only when neither non-cognate orientation,
A with reverse-complement(B) or B with reverse-complement(A), binds more
strongly than the threshold against any strand already in that bucket.
Consequently, every bucket is an orthogonal set and no exhaustive graph search
is needed.
"""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import ViennaRNA as RNA


DEFAULT_BUCKET_COUNT = 1
DEFAULT_CLIQUE_SIZE = 4
DEFAULT_LENGTH = 12
DEFAULT_MAX_CANDIDATES = 100_000_000
DEFAULT_THRESHOLD_KCAL_PER_MOL = -3.5
DEFAULT_TEMPERATURE_C = 20.0
DEFAULT_SALT_M = 1.0


def has_long_homopolymer(sequence, max_run=2):
    """Return whether a repeated-base run exceeds ``max_run``."""
    run_length = 1
    for previous, current in zip(sequence, sequence[1:]):
        run_length = run_length + 1 if current == previous else 1
        if run_length > max_run:
            return True
    return False


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(str.maketrans("ATCG", "TAGC"))[::-1]


def print_bucket(bucket, clique_size, evaluated_count, restart_count):
    """Print the complete bucket after a strand is accepted."""
    print(
        f"\nBucket updated: {len(bucket)}/{clique_size} strands "
        f"(evaluated={evaluated_count}, restarts={restart_count})"
    )
    for index, sequence in enumerate(bucket, start=1):
        print(f"{index}: {sequence}")


def fold(sequence, temperature_C, salt_M):
    """Return the MFE structure and energy for one sequence or complex."""
    md = RNA.md()
    md.temperature = float(temperature_C)
    md.salt = float(salt_M)
    md.dangles = 2
    structure, mfe = RNA.fold_compound(sequence, md).mfe()
    return structure, float(mfe)


def interaction_delta_g(
    sequence_a,
    sequence_b,
    monomer_mfes,
    temperature_C,
    salt_M,
):
    """Return the worst unintended interaction between two sequence pairs.

    A-B, RC(A)-RC(B), A-RC(B), and B-RC(A) are evaluated. Intended cognate
    A-RC(A) and B-RC(B) binding is deliberately excluded.
    """
    sequence_a_rc = reverse_complement(sequence_a)
    sequence_b_rc = reverse_complement(sequence_b)
    unintended_energies = []
    for strand_a, strand_b in (
        (sequence_a, sequence_b),
        (sequence_a_rc, sequence_b_rc),
        (sequence_a, sequence_b_rc),
        (sequence_b, sequence_a_rc),
    ):
        _, dimer_mfe = fold(
            f"{strand_a}&{strand_b}",
            temperature_C,
            salt_M,
        )
        cofold_delta_g = (
            dimer_mfe
            - monomer_mfes[strand_a]
            - monomer_mfes[strand_b]
        )
        unintended_energies.append(min(0.0, cofold_delta_g))
    return min(unintended_energies)


def self_interaction_delta_g(
    sequence,
    monomer_mfes,
    temperature_C,
    salt_M,
):
    """Return the worse of sequence and reverse-complement homodimerization."""
    sequence_rc = reverse_complement(sequence)
    homodimer_energies = []
    for strand in (sequence, sequence_rc):
        _, dimer_mfe = fold(
            f"{strand}&{strand}",
            temperature_C,
            salt_M,
        )
        cofold_delta_g = dimer_mfe - 2.0 * monomer_mfes[strand]
        homodimer_energies.append(min(0.0, cofold_delta_g))
    return min(homodimer_energies)


def directional_binding_delta_g(
    strand,
    complement,
    temperature_C,
    salt_M,
):
    """Return effective binding Delta G for one strand/complement orientation."""
    _, strand_mfe = fold(strand, temperature_C, salt_M)
    _, complement_mfe = fold(complement, temperature_C, salt_M)
    _, dimer_mfe = fold(
        f"{strand}&{complement}",
        temperature_C,
        salt_M,
    )
    return min(0.0, dimer_mfe - strand_mfe - complement_mfe)


def display_interaction_heatmap(
    strands,
    temperature_C,
    salt_M,
    workers=None,
):
    """Display a 2N-by-2N strand and reverse-complement interaction matrix."""
    complements = [reverse_complement(strand) for strand in strands]
    all_sequences = list(strands) + complements
    strand_count = len(strands)
    tasks = [
        (row, column, all_sequences[row], all_sequences[column])
        for row, column in combinations_with_replacement(
            range(len(all_sequences)),
            2,
        )
    ]
    worker_count = workers or min(32, (os.cpu_count() or 1) + 4)

    def evaluate(task):
        row, column, sequence_a, sequence_b = task
        delta_g = directional_binding_delta_g(
            sequence_a,
            sequence_b,
            temperature_C,
            salt_M,
        )
        return row, column, delta_g

    matrix = np.empty((len(all_sequences), len(all_sequences)), dtype=float)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for row, column, delta_g in executor.map(evaluate, tasks):
            matrix[row, column] = delta_g
            matrix[column, row] = delta_g

    size = max(8.0, 0.55 * len(all_sequences))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="viridis", aspect="equal")
    figure.colorbar(image, ax=axis, label="Effective binding ΔG (kcal/mol)")
    labels = (
        [f"S{index + 1}" for index in range(strand_count)]
        + [f"RC{index + 1}" for index in range(strand_count)]
    )
    axis.set_xticks(range(len(all_sequences)), labels=labels, rotation=90)
    axis.set_yticks(range(len(all_sequences)), labels=labels)
    axis.set_xlabel("Sequence")
    axis.set_ylabel("Sequence")
    axis.set_title("2N × 2N strand/reverse-complement interaction energies")

    midpoint = (float(matrix.min()) + float(matrix.max())) / 2.0
    for row in range(len(all_sequences)):
        for column in range(len(all_sequences)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < midpoint else "black",
            )

    figure.tight_layout()
    plt.show()
    return matrix


def next_unstructured_sequence(rng, length, seen, monomer_mfes, temperature_C, salt_M, max_attempts=int(1e5), three_letter=False):
    """Generate a strand whose monomer and reverse complement are unpaired."""
    for _ in range(max_attempts):
        alphabet = "ATCG"
        if three_letter:
            alphabet = "ATC"
        sequence = "".join(rng.choice(alphabet) for _ in range(length))
        if sequence in seen or has_long_homopolymer(sequence):
            continue
        seen.add(sequence)
        sequence_rc = reverse_complement(sequence)
        structure, mfe = fold(sequence, temperature_C, salt_M)
        rc_structure, rc_mfe = fold(sequence_rc, temperature_C, salt_M)
        if structure == "." * length and rc_structure == "." * length:
            monomer_mfes[sequence] = mfe
            monomer_mfes[sequence_rc] = rc_mfe
            return sequence
    raise RuntimeError(
        f"could not find another unstructured length-{length} strand "
        f"after {max_attempts} attempts"
    )


def register_unstructured_sequence(
    sequence,
    monomer_mfes,
    temperature_C,
    salt_M,
):
    """Cache monomer energies if a strand and its complement are unpaired."""
    if has_long_homopolymer(sequence):
        return False
    sequence_rc = reverse_complement(sequence)
    structure, mfe = fold(sequence, temperature_C, salt_M)
    rc_structure, rc_mfe = fold(sequence_rc, temperature_C, salt_M)
    if structure != "." * len(sequence) or rc_structure != "." * len(sequence):
        return False
    monomer_mfes[sequence] = mfe
    monomer_mfes[sequence_rc] = rc_mfe
    return True


def grow_orthogonal_buckets(
    bucket_count,
    clique_size,
    length,
    threshold,
    temperature_C,
    salt_M,
    max_candidates,
    seed=None,
    workers=None,
):
    """Grow one bucket using random-restart single-base hill climbing."""
    if bucket_count != 1:
        raise ValueError("mutation-based growth currently requires one bucket")
    if clique_size < 1:
        raise ValueError("clique size must be at least 1")
    if length < 1:
        raise ValueError("strand length must be at least 1")
    if max_candidates < 1:
        raise ValueError("max candidates must be at least 1")

    RNA.params_load_DNA_Mathews2004()
    rng = random.Random(seed)
    seen = set()
    monomer_mfes = {}
    pair_energies = {}
    worker_count = workers or min(32, (os.cpu_count() or 1) + 4)

    evaluated_count = 0
    while True:
        seed_sequence = next_unstructured_sequence(
            rng,
            length,
            seen,
            monomer_mfes,
            temperature_C,
            salt_M,
        )
        evaluated_count += 1
        seed_self_energy = self_interaction_delta_g(
            seed_sequence,
            monomer_mfes,
            temperature_C,
            salt_M,
        )
        pair_energies[(seed_sequence, seed_sequence)] = seed_self_energy
        if seed_self_energy >= threshold:
            break
        if evaluated_count >= max_candidates:
            return [[]], 0, pair_energies, evaluated_count

    bucket = [seed_sequence]
    buckets = [bucket]
    restart_count = 0
    failure_limit = length * 2
    # print_bucket(
    #     bucket,
    #     clique_size,
    #     evaluated_count,
    #     restart_count,
    # )

    if clique_size == 1:
        return buckets, 0, pair_energies, evaluated_count

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        def candidate_score(candidate):
            futures = {}
            self_key = (candidate, candidate)
            if self_key not in pair_energies:
                futures[self_key] = executor.submit(
                    self_interaction_delta_g,
                    candidate,
                    monomer_mfes,
                    temperature_C,
                    salt_M,
                )
            for existing in bucket:
                key = tuple(sorted((candidate, existing)))
                if key not in pair_energies:
                    futures[key] = executor.submit(
                        interaction_delta_g,
                        candidate,
                        existing,
                        monomer_mfes,
                        temperature_C,
                        salt_M,
                    )
            for key, future in futures.items():
                pair_energies[key] = future.result()
            return min(
                pair_energies[self_key],
                *(
                    pair_energies[tuple(sorted((candidate, existing)))]
                    for existing in bucket
                ),
            )

        while evaluated_count < max_candidates:
            candidate = next_unstructured_sequence(
                rng,
                length,
                seen,
                monomer_mfes,
                temperature_C,
                salt_M,
            )
            evaluated_count += 1
            score = candidate_score(candidate)
            failed_changes = 0

            while score < threshold and failed_changes < failure_limit:
                position = rng.randrange(length)
                replacement = rng.choice(
                    [base for base in "ATCG" if base != candidate[position]]
                )
                mutation = (
                    candidate[:position] + replacement + candidate[position + 1:]
                )
                if mutation in seen:
                    failed_changes += 1
                    continue

                seen.add(mutation)
                evaluated_count += 1
                if not register_unstructured_sequence(
                    mutation,
                    monomer_mfes,
                    temperature_C,
                    salt_M,
                ):
                    failed_changes += 1
                    if evaluated_count >= max_candidates:
                        break
                    continue

                mutation_score = candidate_score(mutation)
                if mutation_score > score:
                    candidate = mutation
                    score = mutation_score
                    failed_changes = 0
                else:
                    failed_changes += 1

                if evaluated_count >= max_candidates:
                    break

            if score >= threshold:
                bucket.append(candidate)
                # print_bucket(
                #     bucket,
                #     clique_size,
                #     evaluated_count,
                #     restart_count,
                # )
                if len(bucket) >= clique_size:
                    return buckets, 0, pair_energies, evaluated_count
            else:
                restart_count += 1

    return buckets, 0, pair_energies, evaluated_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incrementally grow buckets of mutually orthogonal DNA strands."
    )
    parser.add_argument(
        "--buckets",
        type=int,
        choices=(1,),
        default=DEFAULT_BUCKET_COUNT,
    )
    parser.add_argument("--clique-size", "-k", type=int, default=DEFAULT_CLIQUE_SIZE)
    parser.add_argument("--length", "-N", type=int, default=DEFAULT_LENGTH)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_KCAL_PER_MOL,
        help="A pair is orthogonal when interaction Delta G >= this value.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_C)
    parser.add_argument("--salt", type=float, default=DEFAULT_SALT_M)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of ViennaRNA interaction-evaluation threads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    buckets, result_index, pair_energies, generated_count = grow_orthogonal_buckets(
        bucket_count=args.buckets,
        clique_size=args.clique_size,
        length=args.length,
        threshold=args.threshold,
        temperature_C=args.temperature,
        salt_M=args.salt,
        max_candidates=args.max_candidates,
        seed=args.seed,
        workers=args.workers,
    )
    result = buckets[result_index]

    print(f"\nEvaluated {generated_count} candidate strands and mutations.")
    print(f"Bucket sizes: {[len(bucket) for bucket in buckets]}")
    if len(result) >= args.clique_size:
        print(f"Bucket {result_index + 1} reached k={args.clique_size}.")
    else:
        print(
            f"No bucket reached k={args.clique_size}; "
            f"largest was bucket {result_index + 1} with {len(result)} strands."
        )

    if not result:
        print("No strand passed the self-interaction threshold.")
        return

    if len(result) > 1:
        print("Worst pairwise non-cognate cross-binding Delta G values (kcal/mol):")
        minimum = None
        for sequence_a, sequence_b in combinations(result, 2):
            delta_g = pair_energies[tuple(sorted((sequence_a, sequence_b)))]
            print(f"{sequence_a} <-> {sequence_b} = {delta_g:.6g}")
            if minimum is None or delta_g < minimum[0]:
                minimum = (delta_g, sequence_a, sequence_b)
        print(
            f"Minimum interaction Delta G: {minimum[0]:.6g} kcal/mol "
            f"for {minimum[1]} <-> {minimum[2]}"
        )

    display_interaction_heatmap(
        result,
        temperature_C=args.temperature,
        salt_M=args.salt,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

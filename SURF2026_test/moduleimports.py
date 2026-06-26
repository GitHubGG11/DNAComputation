import ViennaRNA as RNA
from matplotlib import pyplot as plt
import numpy as np
import random, copy, time
from functools import lru_cache
import math
import itertools
from alive_progress import alive_bar
# SURF2026_test/services.py

import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Tell Django where settings.py is
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SURF2026.settings")

import django
django.setup()

# Now import your Django model/function
from SURF2026_test.models import replace_sequence_results

RNA.params_load_DNA_Mathews2004() 

RNA.cvar.temperature=20  
RNA.cvar.salt=1
RNA.cvar.dangles=2

linker = 'TATCGATA'

loop = "ATC"
bp = {"A": "T", "T": "A", "C": "G", "G": "C"}

seq1 = "GTATT"
seq2 = "CATAA"[::-1]

N = len(seq1)
iterations = 100000


sequences = []


def eval_fitness(strand1, strand2, temp1, temp2):
    RNA.cvar.temperature = temp1
    md = RNA.md()
    compound = RNA.fold_compound(strand1+'&'+strand2, md)
    G1 = compound.mfe()[-1]

    RNA.cvar.temperature = temp2
    md = RNA.md()
    compound2 = RNA.fold_compound(strand1+'&'+strand2, md)
    G2 = compound2.mfe()[-1]
    return G2 - G1


def random_sequence(sequence1, sequence2, mut_end, mut_start=0):
    new_sequence = copy.deepcopy(sequence1)
    pos = random.randint(mut_start, mut_end-1)
    new_bp = random.choice([p for p in bp if p != new_sequence[pos]])
    new_sequence = new_sequence[:pos] + new_bp + new_sequence[pos+1:]
    new_sequence2 = sequence2[:pos] + bp[new_bp] + sequence2[pos+1:]

    return new_sequence, new_sequence2


def dissociation(sequence1, sequence2, temp):
    # why is partition function negative????

    RNA.cvar.temperature = temp
    md = RNA.md()
    pf1 = RNA.fold_compound(sequence1, md).pf()[-1] + RNA.fold_compound(sequence2, md).pf()[-1]
    pf1 = np.exp(-pf1/(0.0019872041*(temp + 273)))

    md = RNA.md()
    pf2 = RNA.fold_compound(sequence1 + '&' + sequence2, md).pf()[-1]
    pf2 = np.exp(-pf2/(0.0019872041*(temp + 273)))

    return pf1 / pf2

def brute_force(temp1, temp2):
    combinations = itertools.product(list(bp.keys()), repeat=5)
    all_combinations = [''.join(combo) for combo in combinations]

    allowed = []
    N = len(all_combinations)
    found = 0
    with alive_bar(N**2) as bar:
        for strand1 in all_combinations:
            for strand2 in all_combinations:
                diff = eval_fitness(strand1, strand2, temp1, temp2)
                attachment = dissociation(strand1, strand2, temp2) > (1-1e-2) and dissociation(strand1, strand2, temp1) < 1e-3
                if attachment:
                    allowed.append([(strand1, strand2), diff])
                    # print("yay", allowed)
                    found += 1
                bar.text = f'Found {found} sequences!'
                bar()
    
    return allowed

def hill_climb(iterations, temp1, temp2, seq1, seq2):

    # TODO: Make sure that we get a list, not just the best

    most = 0
    best_seq1, best_seq2 = "", ""

    for i in range(iterations):

        diff1 = eval_fitness(seq1, seq2, temp1, temp2)

        new_seq1, new_seq2 = random_sequence(seq1, seq2, N)
        diff2 = eval_fitness(new_seq2, new_seq1, temp1, temp2)

        if random.random() < 0.1:
            new_seq2 = random_sequence(seq2, seq1, N)[0]

        change = random.random() < np.exp(2*min(diff2 - diff1, 0))

        if change:
            seq1, seq2 = new_seq1, new_seq2
            attachment = dissociation(new_seq1, new_seq2, temp2) > 0.8 and dissociation(new_seq1, new_seq2, temp1) < 1e-3
            if diff2 > most and attachment:
                best_seq1, best_seq2 = new_seq1, new_seq2
                most = max(most, diff2)

            if attachment:
                sequences.append([(new_seq1, new_seq2), diff2])

    return best_seq1, best_seq2

replace_sequence_results('screening', brute_force(20, 100))



strands = ['CATCATCA',
 'ATTTATTT',
 'CTCTCTCA',
 'CTTTCTAA',
 'TTCCCTAC',
 'TACCTCCT',
 'CTACCCAA',
 'TAATTTAA',
 'CTTCTTCA',
 'CACACACT',
 'TATTATAT',
 'CTAACTAA',
 'TTTATTAA',
 'CAACAACT',
 'AATATATA',
 'TCTTACTT',
 'TGATGATG',
 'AAATAAAT',
 'TGAGAGAG',
 'TTAGAAAG',
 'GTAGGGAA',
 'AGGAGGTA',
 'TTGGGTAG',
 'TTAAATTA',
 'TGAAGAAG',
 'AGTGTGTG',
 'ATATAATA',
 'TTAGTTAG',
 'TTAATAAA',
 'AGTTGTTG',
 'TATATATT',
 'AAGTAAGA']


# print(dissociation(seq1, seq2, 370), RNA.fold_compound("AC" + '&' + "TG", RNA.md()).pf(), RNA.fold_compound("AACC" + '&' + "TTGG"[::-1], RNA.md()).mfe())
# quit()

# strand 1 = (3' to 5', sorry) AC (loop) CATCATCA (domain from hill climbing) AACA (toehold) T CGCGC (falls apart)
# strand 1 (5' to 3') = CGCGC (falls apart) T AACA (toehold) CATCATCA (domain from hill climbing) AC (loop) 
# strand 2 (5' to 3') = GCAAG (falls apart) T AACA (toehold) TCTTACTT (domain from hill climbing) AC (loop)

# still sticks together
# 
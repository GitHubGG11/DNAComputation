import sys, pkgutil, subprocess
import ViennaRNA as RNA
from matplotlib import pyplot as plt
import numpy as np
import random, copy, time
from functools import lru_cache
import math
import itertools

RNA.params_load_DNA_Mathews2004() 

RNA.cvar.temperature=20  
RNA.cvar.salt=1
RNA.cvar.dangles=2

loop = "ATC"
bp = {"A": "T", "T": "A", "C": "G", "G": "C"}

seq1 = "ATTGA"
seq2 = "TAACT"[::-1]

N = len(seq1)
iterations = 10000
most = 0
best_seq1, best_seq2 = "", ""


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
    pf1 = np.exp(-pf1/(0.0019872041*(temp1 + 273)))

    md = RNA.md()
    pf2 = RNA.fold_compound(sequence1 + '&' + sequence2, md).pf()[-1]
    pf2 = np.exp(-pf2/(0.0019872041*(temp1 + 273)))

    return pf1 / pf2


for i in range(iterations):
    temp1 = 20
    temp2 = 100

    diff1 = eval_fitness(seq1, seq2, temp1, temp2)

    new_seq1, new_seq2 = random_sequence(seq1, seq2, N)
    diff2 = eval_fitness(new_seq2, new_seq1, temp1, temp2)

    if random.random() < 0.1:
        new_seq2 = random_sequence(seq2, seq1, N)[0]

    change = random.random() < np.exp(3*min(diff2 - diff1, 0))

    if change:
        seq1, seq2 = new_seq1, new_seq2
        attachment = dissociation(new_seq1, new_seq2, temp2) > 0.5 and dissociation(new_seq1, new_seq2, temp1) < 1e-3
        if diff2 > most and attachment:
            best_seq1, best_seq2 = new_seq1, new_seq2
            most = max(most, diff2)

        print(seq1, seq2, most, dissociation(new_seq1, new_seq2, temp2), dissociation(new_seq1, new_seq2, temp1))

print(best_seq1, best_seq2)


# print(dissociation(seq1, seq2, 370), RNA.fold_compound("AC" + '&' + "TG", RNA.md()).pf(), RNA.fold_compound("AACC" + '&' + "TTGG"[::-1], RNA.md()).mfe())
# quit()
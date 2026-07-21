import ViennaRNA as RNA
from delta_g_estimator import reverse_complement_dna


linker = 'TATCGATA'

arm1 = 'GCGTCCGACACTGAAC'
arm2 = 'TGTCAGCCGTGCTATC'
arm3 = 'ACGGTCTGACCCGAAA'
arm4 = 'ATGTGGCCTACAGTGA'

arm4c = reverse_complement_dna(arm4)
arm1c = reverse_complement_dna(arm1)
arm2c = reverse_complement_dna(arm2)
arm3c = reverse_complement_dna(arm3)

RNA.params_load_DNA_Mathews2004()




def nsEnergyDiff(domainA='ACGTTGCAAGTC', linker='GGTA', G='CG', temp=30, middle="GTACT"):
    domainAc = reverse_complement_dna(domainA)

    armL = len(arm1)

    domainL = len(domainA)       # 12
    bulgeL = len(G)          # 2
    toeholdL = len(linker)      # 4

    ideal_parts = [
        "(" * domainL + "(" * armL + ".." + "(" * armL + "." * bulgeL + "(" * domainL + "." * (toeholdL + 1),

        ")" * domainL + ")" * armL + ".." + "(" * armL + "." * bulgeL + "(" * domainL + "." * (toeholdL + 1),

        ")" * domainL + ")" * armL + ".." + "(" * armL + "." * bulgeL + "(" * domainL + "." * (toeholdL + 1),

        ")" * domainL + ")" * armL + ".." + ")" * armL + "." * bulgeL + ")" * domainL + "." * (toeholdL + 1),
    ]

    ideal_parts_linked = [
        "(" * domainL
        + "(" * armL
        + ".."
        + "(" * armL
        + "." * bulgeL
        + "(" * domainL
        + "." * (toeholdL + 1),

        ")" * domainL
        + ")" * armL
        + ".."
        + "(" * armL
        + "." * bulgeL
        + "(" * domainL
        + "." * (toeholdL + 1),

        ")" * domainL
        + ")" * armL
        + ".."
        + "(" * armL
        + "." * bulgeL
        + "(" * domainL
        + "." * (toeholdL + 1),

        ")" * domainL
        + ")" * armL
        + ".."
        + ")" * armL
        + "." * bulgeL
        + ")" * domainL
        + "." * (toeholdL + 1),
    ]


    ideal_parts = "".join(ideal_parts)
    ideal_parts_linked = "".join(ideal_parts_linked)


    s1 = domainA + arm4c + "TT" + arm1 + G + domainAc + "T" + linker
    s2 = domainA + arm1c + "TT" + arm2 + G + domainAc + "T" + linker
    s3 = domainA + arm2c + "TT" + arm3 + G + domainAc + "T" + linker
    s4 = domainA + arm3c + "TT" + arm4 + G + domainAc + "T" + linker
    
    linkerc = reverse_complement_dna(linker)

    strandLock = middle + "T" + linkerc + domainA + reverse_complement_dna(G)

    middlec = reverse_complement_dna(middle)

    strandLockc = middlec + "T" + linkerc + domainA + reverse_complement_dna(G)

    md = RNA.md()
    md.temperature = temp
    md.salt = 1.0
    md.dangles = 2

    ns = [s1, s2, s3, s4]

    sequence = "&".join(ns + [strandLock, strandLockc] + ns)
    compound = RNA.fold_compound(sequence, md)
    structure, mfe = compound.mfe()

    md = RNA.md()
    md.temperature = temp
    md.salt = 1.0
    md.dangles = 2

    sequence_null = "&".join(ns + ns)
    compound_null = RNA.fold_compound(sequence_null, md)
    mfe_null = compound_null.eval_structure(ideal_parts + ideal_parts)

    print(structure)
    print("\n")
    print(ideal_parts_linked)

    RNA.svg_rna_plot(sequence_null.replace("&", ""), compound_null.mfe()[0], "ideal_nanostar.svg")

    return mfe - mfe_null

mfe_diff = nsEnergyDiff(temp=0)

print(f"MFE: {mfe_diff:.2f} kcal/mol")
print("Matches ideal nanostar:")
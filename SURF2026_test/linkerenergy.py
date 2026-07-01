import ViennaRNA as RNA
from moduleimports import dissociation

seq = 'ACGCG'
seqcom = seq
domainA = 'TATCGATA'
domainB = 'TCATATGA'
linker = 'ATAT'
g = 'CG'

solute = [0.3, 0.4]
solvent = 1-sum(solute)
Z = 4

strand1 = g + domainA + linker + 'T' + seq

import sys
import os
from Bio import SeqIO
import warnings

warnings.filterwarnings('ignore')

os.chdir(r'C:\\Users\\18000\\Desktop\\rand_init')
pdb_list = os.listdir(r'C:\\Users\\18000\\Desktop\\rand_init')

for pdb_file in pdb_list:
    with open(pdb_file, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            print(record.seq)




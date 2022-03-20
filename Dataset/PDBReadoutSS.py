import os
import re
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP, ss_to_index, dssp_dict_from_pdb_file
# from multiprocessing.dummy import Pool

os.chdir(
    'D:\文件\北大\MDL\ProteinBackboneDesign\Dataset'
)
pdb_file = '1US0_A.pdb'



p = PDBParser(PERMISSIVE=1)
model = p.get_structure(pdb_file[:4], pdb_file)[0]
chain = model[pdb_file[5]]

dssp = DSSP(model, pdb_file)
dssp_keys = list(dssp.keys())

print(dssp_keys)

chain_len = 0
pos_list = []
ss_list = []


for res in chain.get_residues():
    print(res)
    if res.id[0] == ' ':
        chain_len += 1
        print(chain_len)

        # 1. obtain secondary structure type
        ss_type = dssp[dssp_keys[chain_len-1]][2]

        print(ss_type)
            
        if ss_type == ('H' or 'G' or 'I'):
            ss_list.append(0)
        elif ss_type == 'E':
            ss_list.append(1)
        else:
            ss_list.append(2)


# Train | find 8 bad cases:
# 1US0_A.pdb
# 1X6Z_A.pdb
# 2YKZ_A.pdb
# 3ZOJ_A.pdb
# 4PSS_A.pdb
# 4REK_A.pdb
# 4Y9W_A.pdb
# 6EIO_A.pdb

# problem: residue numbers do not start from 1


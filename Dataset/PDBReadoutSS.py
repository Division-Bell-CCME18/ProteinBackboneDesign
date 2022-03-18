from itertools import chain
import os
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

chain_len = 0
pos_list = []
ss_list = []


for res in chain.get_residues():
    if res.id[0] == ' ':
        chain_len += 1

        # 1. obtain secondary structure type
        ss_type = dssp[dssp_keys[chain_len-1]][2]
            
        if ss_type == ('H' or 'G' or 'I'):
            ss_list.append(0)
        elif ss_type == 'E':
            ss_list.append(1)
        else:
            ss_list.append(2)


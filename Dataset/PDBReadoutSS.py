import os
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP, ss_to_index, dssp_dict_from_pdb_file
# from multiprocessing.dummy import Pool

dir = 'D:\文件\北大\github\ProteinBackboneDesign\Dataset'
file_name = '1NWZ_A.pdb'

# def get_dssp_dict(pdb_file):


p = PDBParser()
structure = p.get_structure(file_name[:4], dir+'\\'+file_name)
model = structure[0]
dssp = DSSP(model, dir+'\\'+file_name)

a_key = list(dssp.keys())
for i in a_key:
    print(ss_to_index(dssp[i][2]))


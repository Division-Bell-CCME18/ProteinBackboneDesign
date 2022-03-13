import os
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from multiprocessing.dummy import Pool

dir = 'D:\文件\北大\github\ProteinBackboneDesign\Dataset\PDBDataset_raw_chain'
list = os.listdir(dir)
mis_list = []

def read_ss(file_name):
    p = PDBParser()
    structure = p.get_structure(file_name[:4], dir+'\\'+file_name)
    model = structure[0]
    try:
        dssp = DSSP(model, dir+'\\'+file_name)
    except:
        mis_list.append(file_name)



with Pool(15) as pool:
    pool.map(read_ss, list)

print(mis_list)







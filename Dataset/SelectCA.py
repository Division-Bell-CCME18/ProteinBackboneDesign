import os
from multiprocessing.dummy import Pool


os.chdir(
    'D:\文件\北大\github\ProteinBackboneDesign\Dataset\PDBDataset_raw_chain'
)

chain_list = []

for i in os.listdir('D:\文件\北大\github\ProteinBackboneDesign\Dataset\PDBDataset_raw_chain'):
    chain_list.append(i[:6])


def select(chain_id):
    os.system(
        f'pdb_selatom -CA {chain_id}.pdb > D:\文件\北大\github\ProteinBackboneDesign\Dataset\PDBDataset_CA\{chain_id}_CA.pdb'
    )

with Pool(15) as p:
    p.map(select, chain_list)
    

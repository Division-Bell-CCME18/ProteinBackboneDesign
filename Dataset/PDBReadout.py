import os
from multiprocessing.dummy import Pool

os.chdir(
    'D:\文件\北大\github\ProteinBackboneDesign\Dataset'
)


def process(lines):
    PDB = lines.split()[0]
    PDBID = PDB[:4]
    chain = PDB[4:] 
    if len(chain) == 1:
        os.system(
            'pdb_fetch %s > %s.pdb' % (PDBID, PDBID)
        )
        os.system(
            'pdb_selchain -%s %s.pdb > PDBDataset_raw_chain/%s_%s.pdb' % (chain, PDBID, PDBID, chain)
        )



with open('PDBList.txt', 'r') as file_to_read:    
    os.chdir(
        'PDBDataset_raw'
    )
    with Pool(15) as p:
        p.map(process, file_to_read)


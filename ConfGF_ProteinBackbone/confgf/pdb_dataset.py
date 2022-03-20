import os
import copy
from tqdm import tqdm

import numpy as np
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from torch_scatter import scatter

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities


dir = 'D:\文件\北大\MDL\ProteinBackboneDesign\Dataset'
pdb_file = '1US0_A.pdb'


def set_working_dir(dir):
    """
    set default working directory
    """
    os.chdir(
        dir
    )


# set_working_dir(dir)


def pdb_to_data(pdb_file):
    """
    Covert a .pdb file to a pyg object that can be fed into GNN
    """
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


            # 2. obtain C-alpha position
            atom_CA = res['CA']
            pos_list.append(list(atom_CA.coord))


    # 3. construct edges
    edge_list = []


    # i) coordinate-based neighbors
    # neighbor search, radius in angstrom (to be decided)
    atom_list = unfold_entities(chain, 'A')
    neighbor_search = NeighborSearch(atom_list)
    contact_list = neighbor_search.search_all(radius=5, level='R')

    for pair in contact_list:
        # exclude water and HETATM
        res_1, res_2 = pair[0].id, pair[1].id
        if (res_1[0] == ' ') and (res_2[0] == ' '):
            edge_list.append([res_1[1], res_2[1]])
            edge_list.append([res_2[1], res_1[1]])


    # ii) sequence-based neighbors 
    # 2^m sequence separation
    for i in range(1, chain_len+1):
        for j in range(i+1, chain_len+1):
            if ((j-i) & (j-i-1) == 0) and ([i, j] not in edge_list):
                edge_list.append([i, j])
                edge_list.append([j, i])


    # iii) random neighbors (rand-size to be decided)
    k = 0
    rand_size = 0.01
    while k <= int(chain_len ** 2 * rand_size):
        (rand_1, rand_2) = (np.random.randint(1, chain_len+1), np.random.randint(1, chain_len+1))
        if (rand_1 != rand_2) and ([rand_1, rand_2] not in edge_list):
            edge_list.append([rand_1, rand_2])
            edge_list.append([rand_2, rand_1])
            k += 1
        

    node_feature = torch.tensor(ss_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float32)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    gragh_label = pdb_file[:6]

    data = Data(x=node_feature, edge_index=edge_index, pos=pos, y=gragh_label)

    return data
    

# print(pdb_to_data(pdb_file))





dataset_dir = 'D:\文件\北大\MDL\ProteinBackboneDesign\Dataset\PDBDataset_test'

def preprocess_pdb_dataset(dataset_dir=dataset_dir):
    """
    preprocess pdb dataset
    """
    set_working_dir(dataset_dir)
    pdb_list = os.listdir(dataset_dir)

    print('process train...')
    all_train = []
    bad_case = []
    for i in tqdm(range(len(pdb_list))):
        pdb = pdb_list[i]
        try:
            data = pdb_to_data(pdb)
            all_train.append(data)
        except:
            bad_case.append(pdb)


    print('Train | find %d pdb files as training data' % len(all_train))
    print('Train | find %d bad cases:' % len(bad_case))
    for i in bad_case:
        print(i)

    return all_train

preprocess_pdb_dataset(dataset_dir=dataset_dir)






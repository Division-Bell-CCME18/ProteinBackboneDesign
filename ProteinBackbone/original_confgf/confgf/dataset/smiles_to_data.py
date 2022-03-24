import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx
from torch_scatter import scatter
#from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

from confgf import utils


def smiles_to_data(smiles):
    """
    Convert a SMILES to a pyg object that can be fed into ConfGF for generation
    """
    try:    
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    except:
        return None
        
    N = mol.GetNumAtoms()
    pos = torch.rand((N, 3), dtype=torch.float32)

    atomic_number = []
    aromatic = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    print(row)
    print(col)
    print(edge_type)


    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)

    # print(edge_index)
    
    transform = Compose([
        utils.AddHigherOrderEdges(order=3),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])
    
    print(data)
    print(transform(data))


smiles_to_data('CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H](C2(C)C)/C=C(\C)/C(=O)OC)C/C=C\C=C')

from ctypes import Structure
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


def set_dir(dir):
    os.chdir(
        dir
    )



def pdb_to_data(pdb_file):
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_file[:4], pdb_file)
    model = structure[0]
    chain = model[pdb_file[5]]
    





def process_pdb_dataset():
    return 0
    
    




    







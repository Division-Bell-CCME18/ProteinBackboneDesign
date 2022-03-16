import os
import copy
from tqdm import tqdm

import numpy as np
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from torch_scatter import scatter


def set_dir(dir):
    os.chdir(
        dir
    )

def get_num_aa(pdb_file):
    # aa: amino acid
    return 0
    
def get_ss(pdb_file):
    return 0

def get_pos(pdb_file):
    return 0



def pdb_to_data(pdb_file):
    get_ss(pdb_file)
    get_pos(pdb_file)


def process_pdb_dataset():
    
    




    







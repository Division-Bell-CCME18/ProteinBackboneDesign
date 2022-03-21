import os
from tqdm import tqdm
import warnings
import pickle
import argparse

import numpy as np

import torch
from torch_geometric.data import Data


from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split the pdb dataset into training set and test set')
    parser.add_argument('--pickle_dir', type=str, default=os.getcwd())
    parser.add_argument('--train_dir', type=str, default=os.getcwd())
    parser.add_argument('--test_dir', type=str, default=os.getcwd())
    parser.add_argument('--train_size', type=float, default=0.9)

    args = parser.parse_args()
    pickle_dir = args.pickle_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    train_size = args.train_size

    train_data, test_data = [], []
    test_size = 1. - train_size

    # generate train / test split indexes
    



    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    pickle_dir = args.pickle_dir




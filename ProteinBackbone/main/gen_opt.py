#coding: utf-8

import os
import sys
import argparse
from easydict import EasyDict
import yaml
import random

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

from pdb_utils import pdb_to_data, update_pdb_info, gen_perturb
from network_utils import runner, scorenet, transforms



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate the model performance via optimization of given protein backbone structure')

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--pdb_id', type=str, default='all')
    parser.add_argument('--sigma_perturb', type=float, default=0.2)
    parser.add_argument('--steps_pos', type=int, default=None)
    parser.add_argument('--input_perturb', type=int, default=False)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2022:
        config.train.seed = args.seed

    if config.test.output_path is not None:
        config.test.output_path = os.path.join(config.test.output_path, config.model.name)
        if not os.path.exists(config.test.output_path):
            os.makedirs(config.test.output_path)

    os.chdir(args.working_dir)
    log_file = open('gen_opt.log', mode='w', encoding='utf-8')
    sys.stdout = log_file

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)    
    config.train.device = device
    config.train.gpus = gpus

    print(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)        
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch \n')


    def gen_opt(pdb_id, working_dir, config):
        """
        generate optimized backbone structure of given protein 
        """

        # convert pdb file
        print(f'loading pdb file (id: %s) from %s as test data...' % (pdb_id, working_dir))

        






    log_file.close()


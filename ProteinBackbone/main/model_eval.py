#coding: utf-8

import os
import argparse
from easydict import EasyDict
import yaml
import random

import numpy as np

import torch
from torch_geometric.data import Data

from pdb_utils import pdb_to_data, update_pdb_info, gen_perturb
from network_utils import runner, scorenet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate the model performance via optimization of given protein backbone structure')

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--pdb_id', type=str, required=True)
    parser.add_argument('--sigma_perturb', type=float, default=1.0)

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
    print('set seed for random, numpy and torch')

    # convert pdb file
    os.chdir(args.working_dir)
    print(f'loading pdb file (id: %s) from %s as test data' % (args.pdb_id, args.working_dir))

    train_data = []
    val_data = []
    test_data = []

    data = pdb_to_data(args.pdb_id + '.pdb')
    perturb_data = gen_perturb(data, sigma_perturb=args.sigma_perturb, sigma_end=config.model.sigma_end)

    test_data.append(perturb_data)
    
    print('train size : %d  ||  val size: %d  ||  test size: %d ' % (len(train_data), len(val_data), len(test_data)))
    print('loading test data done!')


    # preprocess original pdb file (select CA and remove HETATM)
    os.system(
        f'pdb_delhetatm {args.pdb_id}.pdb | pdb_selatom -CA | pdb_tidy > {args.pdb_id}_tidy.pdb'
    )


    model = scorenet.DistanceScoreMatch(config)

    #optimizer = utils.get_optimizer(config.train.optimizer, model)    
    optimizer = None
    #scheduler = utils.get_scheduler(config.train.scheduler, optimizer)
    scheduler = None


    solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)

    assert config.test.init_checkpoint is not None
    solver.load(config.test.init_checkpoint, epoch=config.test.epoch)


    # optimize given structure
    return_data = solver.generate_samples_from_data(perturb_data, num_repeat=1, keep_traj=True)

    # write in pdb file
    update_pdb_info(return_data, f'$s_tidy.pdb' % args.pdb_id)



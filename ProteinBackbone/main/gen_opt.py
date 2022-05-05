#coding: utf-8

import os
import sys
import argparse
from easydict import EasyDict
import yaml
import random

import numpy as np

import torch
from torch_geometric.data import Data, Dataset

from pdb_utils import update_pdb_info, extract_sketch_info
from network_utils import runner, scorenet



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate optimized backbone structure of (a single) given protein')

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--pdb_id', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--steps_pos', type=int, default=None)

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


    def gen_opt(sketch_file, pdb_id, working_dir, config, steps_pos=None):
        """
        generate optimized backbone structure of given protein
        """

        # convert pdb file
        print(f'loading pdb file (id: %s) from %s as initial backbone structure...' % (pdb_id, working_dir))

        train_data = []
        val_data = []
        test_data = []

        init_data = extract_sketch_info(sketch_file, pdb_id, working_dir=working_dir)
        print(f'load initial protein backbone structure (C-alpha trace) done! saved as {pdb_id}_CA.pdb')

        model = scorenet.DistanceScoreMatch(config)
        #optimizer = utils.get_optimizer(config.train.optimizer, model)    
        optimizer = None
        #scheduler = utils.get_scheduler(config.train.scheduler, optimizer)
        scheduler = None


        solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)

        assert config.test.init_checkpoint is not None
        solver.load(config.test.init_checkpoint, epoch=config.test.epoch)

        # optimize initial structure
        # pos_gen, _, return_data, pos_traj = solver.ConfGF_generator(data=perturb_data, config=config.test.gen, pos_init=perturb_data.pos) 
        return_data = solver.generate_samples_from_data(init_data, num_repeat=1, keep_traj=True, steps_pos=steps_pos)
        return_data.pos = return_data.pos_gen

        # write in pdb file
        update_pdb_info(return_data, f'{pdb_id}_CA.pdb', save_dir=working_dir, suffix='opt')
        print(f'{pdb_id} backbone structure optimization done! save as {pdb_id}_CA_opt.pdb')



    print('optimization start!')

    try:
        gen_opt(sketch_file=args.sketch_file, pdb_id=args.pdb_id, working_dir=args.working_dir, config=config, steps_pos=args.steps_pos)
        print('optimization of %s succeeded!\n' % args.pdb_id)
    except:
        print('optimization of %s failed!\n' % args.pdb_id)



    log_file.close()


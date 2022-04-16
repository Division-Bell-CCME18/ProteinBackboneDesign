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
from torch_geometric.transforms import Compose

from pdb_utils import pdb_to_data, update_pdb_info, gen_perturb
from network_utils import runner, scorenet, transforms



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate the model performance via optimization of given protein backbone structure')

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--pdb_id', type=str, default='all')
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
    print('set seed for random, numpy and torch')



    def gen_opt(pdb_id, working_dir, sigma_perturb, config):
        """
        generate optimized backbone structure of given protein 
        """

        # convert pdb file
        print(f'loading pdb file (id: %s) from %s as test data...' % (pdb_id, working_dir))

        # preprocess original pdb file (select CA and remove HETATM)
        os.system(
            f'pdb_delhetatm {pdb_id}.pdb | pdb_selatom -CA | pdb_tidy > {pdb_id}_tidy.pdb'
        )

        train_data = []
        val_data = []
        test_data = []

        transform = Compose([
            transforms.AddEdgeLength(),
            transforms.AddPlaceHolder()     
        ])

        data = transform(pdb_to_data(pdb_id + '.pdb'))
        perturb_data = gen_perturb(data, sigma_perturb, sigma_end=config.model.sigma_end)
        update_pdb_info(perturb_data, f'{pdb_id}_tidy.pdb', save_dir=working_dir ,suffix='perturb')
        print(f'save perturbed protein backbone structure done! saved as {pdb_id}_tidy_perturb.pdb')


        test_data.append(perturb_data)

        print('loading test data done!')


        model = scorenet.DistanceScoreMatch(config)

        #optimizer = utils.get_optimizer(config.train.optimizer, model)    
        optimizer = None
        #scheduler = utils.get_scheduler(config.train.scheduler, optimizer)
        scheduler = None


        solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)

        assert config.test.init_checkpoint is not None
        solver.load(config.test.init_checkpoint, epoch=config.test.epoch)


        # optimize given structure
        # pos_gen, _, return_data, pos_traj = solver.ConfGF_generator(data=perturb_data, config=config.test.gen, pos_init=perturb_data.pos) 
        return_data = solver.generate_samples_from_data(perturb_data, num_repeat=1, keep_traj=True)
        return_data.pos = return_data.pos_gen
        
        # write in pdb file
        update_pdb_info(return_data, f'{pdb_id}_tidy.pdb', save_dir=working_dir, suffix='opt')
        print(f'{pdb_id} backbone structure optimization done! save as {pdb_id}_tidy_opt.pdb')




    print('optimization start!')



    if args.pdb_id != 'all':
        print('train size : 0  ||  val size: 0  ||  test size: 1')
        pdb_success = []
        try:
            gen_opt(pdb_id=args.pdb_id, working_dir=args.working_dir, sigma_perturb=args.sigma_perturb, config=config)
            pdb_success.append(args.pdb_id)
        except:
            print(f'optimization of {args.pdb_id} failed!')

    else:
        pdb_list, pdb_success = [], []
        for item in os.listdir(args.working_dir):
            if (item[-4:] == '.pdb') and (len(item) == 10):
                pdb_list.append(item[:6])
        print('train size : 0  ||  val size: 0  ||  test size: %d' % len(pdb_list))
        for pdb_id in pdb_list:
            try:
                gen_opt(pdb_id=pdb_id, working_dir=args.working_dir, sigma_perturb=args.sigma_perturb, config=config)
                pdb_success.append(pdb_id)
            except:
                print(f'optimization of {args.pdb_id} failed!')
        

    print('optimization of %d structures done!' % len(pdb_success))


    log_file.close()


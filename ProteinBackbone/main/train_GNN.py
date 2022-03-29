import os
import pickle
import argparse
import yaml
import random
from easydict import EasyDict

import numpy as np
import torch
from torch_geometric.data import Data, Dataset



from utils import torch_utils, runner, scorenet

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# define dataset object
class PDBDataset(Dataset):

    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.ss_types = self._ss_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.data)

        
    def _ss_types(self):
        """All secondary structure types."""
        ss_types = set([0, 1, 2])
        return sorted(ss_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set([0])
        # NOTE: distinguish connected from disconnected?
        return sorted(edge_types)



# train
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train GNN')
    
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2022:
        config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path)


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

    # load data
    load_path = config.data.base_path
    print('loading data from %s' % load_path)


    train_data = []
    val_data = []
    test_data = []

    if config.data.train_set is not None:          
        with open(os.path.join(load_path, config.data.train_set), "rb") as fin:
            train_data = [Data.from_dict(i) for i in pickle.load(fin)]
    if config.data.val_set is not None:
        with open(os.path.join(load_path, config.data.val_set), "rb") as fin:
            val_data = [Data.from_dict(i) for i in pickle.load(fin)]
    print('train size : %d  ||  val size: %d  ||  test size: %d ' % (len(train_data), len(val_data), len(test_data)))
    print('loading data done!')


    transform = None      
    train_data = PDBDataset(data=train_data, transform=transform)
    val_data = PDBDataset(data=val_data, transform=transform)
    test_data = PDBDataset(data=test_data, transform=transform)

    model = scorenet.DistanceScoreMatch(config)
    optimizer = torch_utils.get_optimizer(config.train.optimizer, model)
    scheduler = torch_utils.get_scheduler(config.train.scheduler, optimizer)

    solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)
    if config.train.resume_train:
        solver.load(config.train.resume_checkpoint, epoch=config.train.resume_epoch, load_optimizer=True, load_scheduler=True)
    solver.train()




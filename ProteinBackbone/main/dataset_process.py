import os
import sys
import argparse

from pdb_utils import process_pdb_dataset




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process the pdb dataset')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--pickle_dir', type=str, default=os.getcwd())
    parser.add_argument('--hbond_threshold', type=float, default=-0.5)
    parser.add_argument('--rsa_threshold', type=float, default=0.2)
    parser.add_argument('--CB_dist_threshold', type=float, default=6)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    pickle_dir = args.pickle_dir

    log_file = open('pdb_dataset.log', mode='w', encoding='utf-8')
    sys.stdout = log_file
    
    process_pdb_dataset(dataset_dir, pickle_dir, hbond_threshold = args.hbond_threshold, rsa_threshold = args.rsa_threshold, CB_dist_threshold = args.CB_dist_threshold)

    log_file.close()



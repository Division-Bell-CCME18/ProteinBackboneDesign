import os
import sys
import argparse

from pdb_utils import process_pdb_dataset




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process the pdb dataset')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--pickle_dir', type=str, default=os.getcwd())
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    pickle_dir = args.pickle_dir

    log_file = open('pdb_dataset.log', mode='w', encoding='utf-8')
    sys.stdout = log_file
    
    process_pdb_dataset(dataset_dir, pickle_dir)

    log_file.close()



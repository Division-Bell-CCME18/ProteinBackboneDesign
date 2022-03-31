import warnings
import sys
import argparse

from utils import filter_pdb_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='filter structures of limited amount of residues from full pdb dataset')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--res_min', type=int, default=30)
    parser.add_argument('--res_max', type=int, default=60)


    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    res_min = args.res_min
    res_max = args.res_max


    log_file = open('dataset_filter.log', mode='w', encoding='utf-8')
    sys.stdout = log_file

    warnings.filterwarnings('ignore')
    filter_pdb_dataset(dataset_dir, save_dir, res_min=res_min, res_max=res_max)
    

    log_file.close()






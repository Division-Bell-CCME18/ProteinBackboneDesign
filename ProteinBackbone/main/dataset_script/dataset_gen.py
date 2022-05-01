import os
from shutil import copy
import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate dataset according to given pdb list (.txt file)')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)



    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    file_name = args.file_name


    pdb_list = os.listdir(dataset_dir)
    gen_list = []

    log_file = open('dataset_gen.log', mode='w', encoding='utf-8')
    sys.stdout = log_file

    with open(file_name, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            pdb_file_name = line.split()[0] + '.pdb'
            if pdb_file_name in pdb_list:
                gen_list.append(line.split()[0])
                try:
                    copy(os.path.join(dataset_dir, pdb_file_name), save_dir)
                except IOError as e:
                    print('unable to copy %s file. %s' % (pdb_file_name, e))
                except:
                    print('unable to copy %s file. unexpected error' % pdb_file_name)


    print('save dataset at %s done! size: %d' % (save_dir, len(gen_list)))
    print('pdb files in generated dataset:')
    for pdb in gen_list:
        print(pdb)



    log_file.close()




import os
import pickle
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='join several pdb datasets in pickle format together')
    parser.add_argument('--dataset_dir', type=str, default=os.getcwd())


    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    joint_dataset = []
    joint_size = 0

    print('load all datasets...')
    for dataset_name in os.listdir(dataset_dir):
        if dataset_name[-3:] == 'pkl':
            with open(os.path.join(dataset_dir, dataset_name), 'rb') as fin:
                dataset = pickle.load(fin)
                joint_dataset += dataset
                joint_size += len(dataset)
            print('find dataset %s! dataset size: %d' % (dataset_name, len(dataset)))


    with open(os.path.join(dataset_dir, 'pdb_dataset_processed.pkl'), 'wb') as fout:
        pickle.dump(joint_dataset, fout)
    print('save joint dataset at %s done! size: %d' % (dataset_dir, joint_size))



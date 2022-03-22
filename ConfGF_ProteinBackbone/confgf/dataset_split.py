import os
import random
import pickle
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split the pdb dataset into training set and test set')
    parser.add_argument('--pickle_dir', type=str, default=os.getcwd())
    parser.add_argument('--train_dir', type=str, default=os.getcwd())
    parser.add_argument('--test_dir', type=str, default=os.getcwd())
    parser.add_argument('--train_ratio', type=float, default=0.9)

    args = parser.parse_args()
    pickle_dir = args.pickle_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    train_ratio = args.train_ratio


    # generate train / test split indexes
    print('load dataset...')
    pickle_path = os.path.join(pickle_dir, 'pdb_dataset_processed.pkl')
    with open(pickle_path, 'rb') as fin:
        all_data = pickle.load(fin)
    print('load full dataset done! dataset size: %d' % len(all_data))

    all_size = len(all_data)
    train_size = int(train_ratio * all_size)
    random.shuffle(all_data)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    with open(os.path.join(train_dir, 'pdb_dataset_train.pkl'), 'wb') as fout:
        pickle.dump(train_data, fout)
    print('save training set at %s done! size: %d' % (train_dir, train_size))

    with open(os.path.join(test_dir, 'pdb_dataset_train.pkl'), 'wb') as fout:
        pickle.dump(test_data, fout)
    print('save test set at %s done! size: %d' % (test_dir, all_size-train_size))







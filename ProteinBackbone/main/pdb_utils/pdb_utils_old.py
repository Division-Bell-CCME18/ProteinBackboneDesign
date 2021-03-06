import os
from tqdm import tqdm
import warnings
import pickle
from shutil import copy

import numpy as np

import torch
from torch_geometric.data import Data


from Bio.PDB import PDBParser, PDBIO, DSSP, NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa


# dir = 'D:\ProteinBackboneDesign\ProteinBackbone\main\pdb_utils'
# pdb_file = '1NWZ_A.pdb'
# pdb_file = 'initial.pdb'

def set_working_dir(dir):
    """
    set default working directory
    """
    os.chdir(
        dir
    )


# set_working_dir(dir)


def pdb_to_data(pdb_file):
    """
    Covert a pdb file to a pyg object that can be fed into GNN
    """
    p = PDBParser(PERMISSIVE=1)
    model = p.get_structure(pdb_file[:4], pdb_file)[0]
    chain = model[pdb_file[5]]

    dssp = DSSP(model, pdb_file)
    dssp_keys = list(dssp.keys())

    # print(dssp_keys)


    chain_len = 0
    pos_list = []
    ss_list = []


    for res in chain.get_residues():
        # print(res.id)
        # also include situations like 1OJH_A.pdb ('H_MSE', 25, ' ') 
        if is_aa(res):
            chain_len += 1

            # 1. obtain secondary structure type
            ss_type = dssp[dssp_keys[chain_len-1]][2]
            
            if ss_type == 'H':
                ss_list.append(0)
            elif ss_type == 'E':
                ss_list.append(1)
            else:
                ss_list.append(2)


            # 2. obtain C-alpha position
            atom_CA = res['CA']
            pos_list.append(list(atom_CA.coord))

    # print(chain_len)

    # avoid residue missing in the middle of the sequence (see 1G3J_B.pdb)
    if chain_len != dssp_keys[-1][1][1] - dssp_keys[0][1][1] + 1:
        raise IndexError('residue index mislabeled!')

    # 3. construct edges
    edge_list = []
    edge_type = []

    
    # i) hydrogen-bond-based neighbors
    threshold = -0.5

    for i in range(0, chain_len):
        for col in [6, 8, 10, 12]:
            hbond_id = int(dssp[dssp_keys[i]][col])
            hbond_energy = float(dssp[dssp_keys[i]][col+1])
            
            # only preserve alpha-alpha, alpha-beta, beta-beta
            if (hbond_energy <= threshold) and (hbond_id != 0) and ([i, i+hbond_id] not in edge_list) and ((i+hbond_id) in range(0, chain_len)) and (dssp[dssp_keys[i]][2] in ['H', 'E']) and (dssp[dssp_keys[i+hbond_id]][2] in ['H', 'E']):
                # print(dssp[dssp_keys[i]][2], dssp[dssp_keys[i+hbond_id]][2])
                edge_list.append([i, i+hbond_id])
                edge_list.append([i+hbond_id, i])
                edge_type += 2 * [0]
                # print([i, i+hbond_id])

    

    # ii) sequence-based neighbors 
    # 2^m sequence separation
    for i in range(0, chain_len):
        for j in range(i, chain_len):
            # if ((j-i) & (j-i-1) == 0) and ([i, j] not in edge_list):
            if ((j-i) == 1) and ([i, j] not in edge_list):
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_type += 2 * [1]
            elif ((j-i) == 2) and ([i, j] not in edge_list):
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_type += 2 * [2]
            # elif ((j-i) == 3) and ([i, j] not in edge_list):
                # edge_list.append([i, j])
                # edge_list.append([j, i])
                # edge_type += 2 * [3]
            # elif ((j-i) == 4) and ([i, j] not in edge_list):
                # edge_list.append([i, j])
                # edge_list.append([j, i])
                # edge_type += 2 * [4]
    



    """
    # ii) coordinate-based neighbors
    # neighbor search, radius in angstrom (to be decided)
    atom_list = unfold_entities(chain, 'A')
    neighbor_search = NeighborSearch(atom_list)
    contact_list = neighbor_search.search_all(radius=2.8, level='R')

    # hbond_list = []

    for pair in contact_list:
        # exclude water and HETATM
        res_1, res_2 = pair[0].id, pair[1].id
        if (res_1[0] == ' ') and (res_2[0] == ' ') and (res_1[1] in range(1, chain_len+1)) and (res_2[1] in range(1, chain_len+1)) and ([res_1[1]-1, res_2[1]-1] not in edge_list):
            edge_list.append([res_1[1]-1, res_2[1]-1])
            edge_list.append([res_2[1]-1, res_1[1]-1])
            edge_type += 2 * [4]
            # hbond_list.append([res_1[1], res_2[1], ss_list[res_1[1]-1], ss_list[res_2[1]-1]])
    
    # print(hbond_list)
    # print(len(hbond_list))

    
    """ 

    """
    # iii) random neighbors (rand-size to be decided)
    k = 0
    rand_size = 0.01
    while k <= int(chain_len ** 2 * rand_size):
        (rand_1, rand_2) = (np.random.randint(1, chain_len+1), np.random.randint(1, chain_len+1))
        if (rand_1 != rand_2) and ([rand_1, rand_2] not in edge_list):
            edge_list.append([rand_1-1, rand_2-1])
            edge_list.append([rand_2-1, rand_1-1])
            edge_type += 2 * [0]
            k += 1
    """

    node_feature = torch.tensor(ss_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float32)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type)
    graph_label = pdb_file[:6]

    data = Data(x=node_feature, edge_index=edge_index, edge_type=edge_type, pos=pos, y=graph_label)

    return data
    

# pdb_to_data(pdb_file)

# print(pdb_to_data(pdb_file))
# print(pdb_to_data(pdb_file).x)
# print(pdb_to_data(pdb_file).edge_index)
# print(pdb_to_data(pdb_file).edge_type)
# print(pdb_to_data(pdb_file).pos)


# dataset_dir = 'D:\ProteinBackboneDesign\Dataset\PDBDataset_test'

def filter_pdb_dataset(dataset_dir, save_dir, res_min=40, res_max=200):
    """
    filter structures of limited amount of residues
    """
    set_working_dir(dataset_dir)
    pdb_list = os.listdir(dataset_dir)

    print('filter structures with %d - %d residues...' % (res_min, res_max))
    filter_list = []

    for i in tqdm(range(len(pdb_list))):
        pdb = pdb_list[i]

        p = PDBParser(PERMISSIVE=1)
        model = p.get_structure(pdb[:4], pdb)[0]
        chain = model[pdb[5]]

        chain_len = 0
        for res in chain.get_residues():
            if res.id[0] == ' ':
                chain_len += 1

        if (chain_len >= res_min) and (chain_len <= res_max):
            filter_list.append(pdb)
            try:
                copy(pdb, save_dir)
            except IOError as e:
                print('unable to copy %s file. %s' % (pdb, e))
            except:
                print('unable to copy %s file. unexpected error' % pdb)

    print('save filtered dataset at %s done! size: %d' % (save_dir, len(filter_list)))
    print('filtered pdb files:')
    for pdb in filter_list:
        print(pdb[:6])

# filter_pdb_dataset(dataset_dir=dataset_dir, save_dir='D:\ProteinBackboneDesign\ProteinBackbone\pdb_dataset\\test')

    


def process_pdb_dataset(dataset_dir, pickle_dir):
    """
    process pdb dataset and save in pickle format
    """
    set_working_dir(dataset_dir)
    pdb_list = os.listdir(dataset_dir)

    print('process dataset...')
    all_data = []
    warning_case = []
    bad_case = []

    warnings.filterwarnings('error')    # catch warnings

    for i in tqdm(range(len(pdb_list))):
        pdb = pdb_list[i]
        try:
            data = pdb_to_data(pdb).to_dict()
            all_data.append(data)
        except Warning:
            warning_case.append(pdb)
        except:
            bad_case.append(pdb)


    print('Process Dataset | find %d pdb files as training data' % len(all_data))
    print('Process Dataset | find %d warning cases:' % len(warning_case))
    for i in warning_case:
        print(i)
    print('Process Dataset | find %d bad cases:' % len(bad_case))
    for j in bad_case:
        print(j)


    with open(os.path.join(pickle_dir, 'pdb_dataset_processed.pkl'), 'wb') as fout:
        pickle.dump(all_data, fout)

    print('save processed dataset done!')


# process_pdb_dataset(dataset_dir=dataset_dir)

# pdb_file = '1NWZ_A.pdb'
# pdb_file_CA ='1NWZ_A_CA_noHET.pdb'
# data = pdb_to_data(pdb_file=pdb_file)
# print(data.pos[0].tolist())


# set_working_dir('D:\ProteinBackboneDesign\ProteinBackbone\main\pdb_utils')

def update_pdb_info(data, pdb_file, save_dir=os.getcwd(), suffix='new'):
    """
    update position information of the original pdb file 
    preprocessed with pdb-tools http://www.bonvinlab.org/pdb-tools/, CA selected and HETATM removed
    """
    set_working_dir(save_dir)


    with open(f'%s_%s.pdb' % (pdb_file[:-4], suffix), 'w') as new_pdb_file:
        with open(pdb_file, 'r') as pdb_file:
            res_list = []
            while True:
                line = pdb_file.readline()
                if not line:
                    break
                elif line.split()[0] != 'ATOM':
                    new_pdb_file.write(line)
                else:
                    res_id = int(line.split()[5])
                    res_list.append(res_id)
                    [x, y, z] = [format(coord, '.3f') for coord in data.pos[res_id-res_list[0]].tolist()]
                    new_pdb_file.write(line[:30]
                        + ' ' * (8-len(str(x))) + str(x) 
                        + ' ' * (8-len(str(y))) + str(y)
                        + ' ' * (8-len(str(z))) + str(z)
                        + line[54:])



# update_pdb_info(data, pdb_file_CA)

# dir = 'D:\ProteinBackboneDesign\ProteinBackbone\main\pdb_utils'
# os.chdir(dir)
# pdb_file = '1NWZ_A.pdb'

# data = pdb_to_data(pdb_file=pdb_file)

def gen_perturb(data, sigma_perturb=1.0, sigma_end=0.01):
    """
    perturb given protein structure with gaussian noise
    """
    pos_init = data.pos

    step_size = 0.00002 * (sigma_perturb / torch.tensor(sigma_end)) ** 2
    noise = torch.randn_like(pos_init) * torch.sqrt(step_size * 2)

    pos = pos_init + noise
    data.pos = pos

    return data


# new_data = gen_perturb(data)
# update_pdb_info(new_data, '1NWZ_A_CA_noHET.pdb')


# sketch_file = 'sketch_extra_info.txt'
# pdb_id = 'initial'


def extract_sketch_info(sketch_file, pdb_id, working_dir=os.getcwd()):
    """
    extract information from initial protein backbone structure generated by SCUBA and return data for subsequent backbone structure optimization
    sketch_file: .txt file that describes secondary structure and hydrogen bond information
    pdb_file: .pdb file generated by SCUBA as initial structure
    """

    set_working_dir(working_dir)

    # pdb file: BB -> CA (C-alpha trace)
    os.system(
        f'pdb_selatom -CA {pdb_id}.pdb | pdb_tidy > {pdb_id}_CA.pdb'
    )

    ss_list, pos_list, edge_list, edge_type = [], [], [], []
 
    # read out C-alpha coordinates and sequence-based neighbors from initial pdb file
    with open(f'{pdb_id}_CA.pdb', 'r') as pdb_file:
        while True:
            line = pdb_file.readline()
            if not line:
                break
            else:
                line_split = line.split()
                if line_split[0] == 'ATOM':
                    [x, y, z] = [float(line_split[6]), float(line_split[7]), float(line_split[8])]
                    pos_list.append([x, y, z])

    chain_len = len(pos_list)

    for i in range(0, chain_len):
        for j in range(i, chain_len):
            if ((j-i) == 1) and ([i, j] not in edge_list):
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_type += 2 * [1]
            elif ((j-i) == 2) and ([i, j] not in edge_list):
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_type += 2 * [2]

    
    # read out secondary structure information and hydrogen-bond-based neighbors from sketch file
    with open(sketch_file, 'r') as sketch_file:
        line_str = ''
        while True:
            line = sketch_file.readline()
            if not line:
                break
            elif line[0] != '#':
                line_str += line
       
        line_block = []
        for block in line_str.split('\n\n'):
            if block != '' and block != '\n':
                line_block.append(block)

        ss_info = line_block[1].split('\n')
        try:
            beta_hb_info = line_block[2].split('\n')
            # print(beta_hb_info)
        except:
            print('no beta-sheet hydrogen bond information found!')
        
    
    chain_len_cnt = 0

    for ss_item in ss_info:
        ss_item = ss_item.split()
        ss_len = int(ss_item[3])
        loop_len = int(ss_item[7])

        if int(ss_item[6]) == 0: # alpha-helix
            for i in range(chain_len_cnt, chain_len_cnt + ss_len):
                for j in range(i, chain_len_cnt + ss_len):
                    if ((j-i) == 4) and ([i, j] not in edge_list):
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                        edge_type += 2 * [0]

            ss_list += ([0] * ss_len)
            ss_list += ([2] * loop_len)
            chain_len_cnt += (ss_len + loop_len)

        elif int(ss_item[6]) == 1: # beta strand
            ss_list += ([1] * ss_len)
            ss_list += ([2] * loop_len)
            chain_len_cnt += (ss_len + loop_len)

    assert chain_len == chain_len_cnt, 'chain length inconsistent!'

    for beta_hb_item in beta_hb_info:
        if beta_hb_item != '':
            [i, j] = [int(beta_hb_item.split()[0])-1, int(beta_hb_item.split()[1])-1]
            if [i, j] not in edge_list:
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_type += 2 * [0]
        

    node_feature = torch.tensor(ss_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float32)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type)
    graph_label = pdb_id

    data = Data(x=node_feature, edge_index=edge_index, edge_type=edge_type, pos=pos, y=graph_label)

    return data

# print(extract_sketch_info(sketch_file, 'initial'))


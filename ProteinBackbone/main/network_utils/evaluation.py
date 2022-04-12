import os
import numpy as np

import torch
from torch_geometric.data import Data

from . import torch_utils



# dir = 'D:\文件\北大\github\ProteinBackboneDesign\ProteinBackbone\main\pdb_utils'
# pdb_file = '1NWZ_A.pdb'



def gen_perturb(data, sigma):
    """
    perturb given protein structure with gaussian noise
    """
    pos_init = data.pos
    d = torch_utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1) # (num_edge, 1)

    return data
# -*- coding: utf-8 -*-
'''
Created on Fri Jan 13 16:45:27 2023

@author: anaso
'''

import os
import torch
from Bio.PDB import PDBParser
import numpy as np
import re
from rdkit import Chem
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.atomic import add_atomic_edges

ele2num = {
    "H": 1,
    "O": 2,
    "N": 3,
    "C": 4,
    "S": 5,
    "SE": 6,
    "P": 7,
    "F": 8,
    "Cl": 9,
    "I": 10,
    "Br": 11
}
num_features = 12

unity_conv = {'mM': -3, 'uM': -6, 'nM': -9, 'pM': -12, 'fM': -15}

# for PL problem
mydir_aff = '../../../datasets/index/INDEX_general_PL_data.2020'
# for PP problem
# mydir_aff = '../../../datasets/index/INDEX_general_PP.2020'


def get_affinities(directory):

    aff_dict = {}
    with open(directory, 'r', encoding='utf-8') as f:
        for line in f:

            if line[0] != '#':
                fields = line.split()
                pdb_id = fields[0]
                log_aff = float(fields[3])
                aff_str = fields[4]
                # for PP problem would be aff_str = fields[3]
                aff_tokens = re.split('[=<>~]+', aff_str)
                assert len(aff_tokens) == 2
                label, aff_unity = aff_tokens
                assert label in ['Kd', 'Ki', 'IC50']
                affinity_value = float(aff_unity[:-2])
                exponent = unity_conv[aff_unity[-2:]]
                # aff_unity
                # list, first element is Kd, Ki or IC50, second is aff
                # first characters contain value (example: 49)
                # last two characters of aff_unity contain unity (example: uM)
                # convert all values of affinity to M
                aff = float(affinity_value) * 10**exponent
                # for PP problem log_aff = float(-np.log10(aff))
                # given pdb_id returns biding type, aff and -log(aff)
                aff_dict[pdb_id] = [label, aff, log_aff]

    return aff_dict


# mydir = '../../../datasets/PP'
mydir = '../../../datasets/refined-set'

# creates a list, where the first element is the id of the compound,
# the second is the location of the pdb file of the corresponding compound


def read_dataset(directory):

    pdb_files = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        files = os.listdir(f)
        pdb_id = filename

        pdb_files += [(pdb_id, os.path.join(f, files[2]),
                       os.path.join(f, files[1]))]

    # for PP case:

#     for filename in os.listdir(directory):

#         pdb_files += [(filename[:4], os.path.join(directory, filename))]

    return pdb_files


# aff_dict = get_affinities(mydir_aff)

def vector2onehot(vector, num_feat):
    onehotvector = np.zeros((len(vector), num_feat))
    onehotvector[np.arange(len(vector)), vector] = 1
    return onehotvector

class PDBDataset(torch.utils.data.Dataset):
    """ for a set of compounds will return their coordinates
    padded and flatten (both the ligand and protein)"""

    def __init__(self, pdb_files, aff_dict):
        '''
        Args:
            pdb_files: list with triplets containing
            name of compound (4 letters)
            path to pdb file describing protein
            path to sdf file describing ligand
        '''

        config = ProteinGraphConfig()
        params_to_change = {
            "granularity": "atom",
            "edge_construction_functions": [add_atomic_edges]
        }
        config = ProteinGraphConfig(**params_to_change)

        self.dataset_len = len(pdb_files)

        # parser = PDBParser(QUIET=True)

        max_len_p = 0
        max_len_l = 0

        data = []

        for comp_name, path_protein, path_ligand in pdb_files:
            # ------------ Protein -------------
            g = construct_graph(config=config, pdb_path=path_protein)
            nodes_dic = {}
            for i, (ident, d) in enumerate(g.nodes(data=True)):
                nodes_dic[ident] = [
                    i, d["element_symbol"], d["coords"]
                ]
            
            nodes_list = list(nodes_dic.values())
            coord_p = [attr[2].tolist() for attr in nodes_list]
            
            
            # structure_pro = parser.get_structure(comp_name, path_protein)
            # coord_p = [atom.get_coord() for atom in structure_pro.get_atoms()]
            
            max_len_p = max(max_len_p, len(coord_p))
            
            elem_p = [attr[1] for attr in nodes_list]
            # elem_p = [atom.element for atom in structure_pro.get_atoms()]
            elem_p_n = [ele2num[i] for i in elem_p]
            
            
            onehotelem_p = vector2onehot(elem_p_n, num_features)
            onehotelem_p[np.arange(len(elem_p_n)), 0] = 1
            
            # ------------ Ligand -------------

            structure_lig = Chem.SDMolSupplier(path_ligand, sanitize=False)[0]
            conf_lig = structure_lig.GetConformer()
            coord_l = conf_lig.GetPositions()
            
            elem_l = [
                atom.GetSymbol() for atom in structure_lig.GetAtoms()
            ]
            
            elem_l_n = [ele2num[i] for i in elem_l]
            onehotelem_l = vector2onehot(elem_l_n, num_features)
            
            
            
            max_len_l = max(max_len_l, len(coord_l))
            

            data += [(torch.as_tensor(np.concatenate((onehotelem_p,coord_p),
                                                     axis=1)), 
                      torch.as_tensor(np.concatenate((onehotelem_l,coord_l),
                                                     axis=1)),
                      aff_dict[comp_name][2])]
            
            
            
        self.data = data
        self.max_len_p = max_len_p
        self.max_len_l = max_len_l

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        coords_p, coords_l, affinity = self.data[index]

        coords_p = torch.nn.functional.pad(
            coords_p, (0, 0, 0, self.max_len_p - coords_p.shape[0]),
            mode='constant',
            value=None)

        coords_l = torch.nn.functional.pad(
            coords_l, (0, 0, 0, self.max_len_l - coords_l.shape[0]),
            mode='constant',
            value=None)

        return torch.flatten(
            torch.cat((coords_p,coords_l), dim = 0).float()), \
            np.float64(affinity)

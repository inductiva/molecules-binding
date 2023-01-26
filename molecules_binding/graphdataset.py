# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:33:28 2023

@author: anaso
"""
from torch_geometric.data import Data, Dataset
# from torch_geometric.loader import DataLoader

import numpy as np
import torch
from rdkit import Chem

from datasets import get_affinities
# from datasets import read_dataset

mydir_aff = "../../../datasets/index/INDEX_general_PL_data.2020"
directory = "../../../datasets/refined-set"

ele2num = {
    "C": 0,
    "H": 1,
    "O": 2,
    "N": 3,
    "S": 4,
    "SE": 5,
    "P": 6,
    "F": 7,
    "Cl": 8,
    "I": 9,
    "Br": 10
}
num_features = len(ele2num)


class GraphDataset(Dataset):

    """
    Args:
        pdb_files: list with triplets containing 
        name of compound (4 letters)
        path to pdb file describing protein
        path to sdf file describing ligand
        
        mydir_aff: to remove later
    """        

    def __init__(self, pdb_files, mydir_aff):

        self.dataset_len = len(pdb_files)

        aff_dict = get_affinities(mydir_aff)

        data_list = []

        for comp_name, path_protein, path_ligand in pdb_files:

            structure_lig = Chem.SDMolSupplier(path_ligand, sanitize=False)[0]
            conf_lig = structure_lig.GetConformer()
            coord_ligand = torch.tensor(conf_lig.GetPositions())

            atoms_ligand_e = [
                atom.GetSymbol() for atom in structure_lig.GetAtoms()
            ]
            atoms_ligand_n = [ele2num[atomtype] for atomtype in atoms_ligand_e]
            atoms_ligand = np.zeros((len(atoms_ligand_n), num_features))

            for i, t in enumerate(atoms_ligand_n):
                atoms_ligand[i, t] = 1.0
            atoms_ligand = torch.tensor(atoms_ligand)

            edges_directed = [[bond.GetBeginAtomIdx(),
                               bond.GetEndAtomIdx()]
                              for bond in structure_lig.GetBonds()]
            edges_bi = []
            for edge in edges_directed:
                i, j = edge
                edges_bi += [[i, j], [j, i]]
            rows = [edge[0] for edge in edges_bi]
            cols = [edge[1] for edge in edges_bi]
            edges = torch.tensor([rows, cols])

            data_list += [(Data(x=atoms_ligand,
                                edge_index=edges,
                                pos=coord_ligand), aff_dict[comp_name][2])]

        self.data_list = data_list

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        graph_ligand, affinity = self.data_list[index]

        return torch.tensor(graph_ligand, affinity)


# Create the dataset object

# pdb_files = read_dataset(directory)
# dataset = GraphDataset(pdb_files, mydir_aff)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

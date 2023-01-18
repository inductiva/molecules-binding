# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:45:27 2023

@author: anaso
"""

import os
import torch 
from Bio.PDB import PDBParser
from findingaffinity import get_affinities
import numpy as np

dataset_dic = {
    'PL': 'datasets/v2020-other-PL',
    'PP': 'datasets/PP',
    'PLrefined': 'datasets/refined-set'
}

# creates a list, where the first element is the id of the compound,
# the second is the location of the pdb file of the corresponding compound


def read_dataset(dataset):

    if dataset not in ['PP', 'PL', 'PLrefined']:
        raise ValueError('argument must be either PP, PL or PLrefined')

    pdb_files = []
    directory = dataset_dic[dataset]

    if dataset == 'PL' or dataset == 'PLrefined':

        for filename in os.listdir(directory):
            #print(filename)
            f = os.path.join(directory, filename)
            files = os.listdir(f)
            pdb_id = files[3][:4]
            pdb_files += [(pdb_id, os.path.join(f, files[3]))]

    if dataset == 'PP':

        for filename in os.listdir(directory):

            pdb_files += [(filename[:4], os.path.join(directory, filename))]

    return pdb_files


aff_dict = get_affinities('PP')


class PDBDataset(torch.utils.data.Dataset):
    
    def __init__(self, pdb_files):
        
        self.pdb_files = pdb_files
        
        parser = PDBParser(QUIET=True)
        self.coordinates = []
        maxlen=0
        self.targets = []
        for pdb_file in pdb_files:
            structure = parser.get_structure(pdb_file[0], pdb_file[1])

            coord = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coord.append(atom.get_coord())
            
            self.targets += [aff_dict[pdb_file[0][:4]][2]]
            
            if len(coord) > maxlen:
                maxlen = len(coord)
            self.coordinates += [torch.tensor(coord)]
            
        for i in range(len(self.coordinates)):
            self.coordinates[i] = torch.nn.functional.pad(
                                    self.coordinates[i], 
                                    (0,0,0, maxlen - 
                                     self.coordinates[i].shape[0]), 
                                    mode='constant', 
                                    value=None)
            #coordinates += [coord.clone().detach().requires_grad_(True)]
        print(maxlen)
    
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, index):
        
        return torch.flatten(self.coordinates[index].float()), \
                            np.float32(self.targets[index])

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:20:23 2023

@author: anaso
"""

import torch 
from Bio.PDB import PDBParser
from findingaffinity import get_affinities
import numpy as np


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
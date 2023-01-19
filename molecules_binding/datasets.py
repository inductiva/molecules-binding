# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:45:27 2023

@author: anaso
"""

import os
import torch 
from Bio.PDB import PDBParser
import numpy as np
import re
from rdkit import Chem

unity_conv = {'mM': -3, 'uM': -6, 'nM': -9, 'pM': -12,'fM': -15}

diri = '../../../datasets'

dataset_dic_aff = {'PL': '/index/INDEX_general_PL_data.2020',
               'PP': '/index/INDEX_general_PP.2020'}

def get_affinities(dataset):
    if dataset not in ['PP','PL']:
        raise ValueError('argument must be either PP or PL')
    
    aff_dict = {}
    with open(diri + dataset_dic_aff[dataset], 'r') as f:
        for line in f:
            
            if line[0]!='#':
                
                fields = line.split()
            
                pdb_id = fields[0]
                
                if dataset == 'PL':
                    log_aff = float(fields[3])
                    aff_str = fields[4]
                    
                elif dataset == 'PP':
                    aff_str = fields[3]
                
                aff_unity = re.split('[=<>~]+', aff_str)
                
                # aff_unity - list, first element is Kd, Ki or IC50, second is aff
                # first characters contain value (example: 49)
                # last two characters of aff_unity contain unity (example: uM)
                
                # convert all values of affinity to M
                aff = float(aff_unity[1][:-2])*10**unity_conv[aff_unity[1][-2:]]
                
                if dataset == 'PP':
                    log_aff = float(-np.log10(aff))
                    
                # given pdb_id returns biding type, aff and -log(aff)
                
                aff_dict[pdb_id] = [aff_unity[0], aff, log_aff]
            
    return aff_dict


dataset_dic = {
    'PLnotrefined': '/v2020-other-PL',
    'PP': '/PP',
    'PL': '/refined-set'
}

# creates a list, where the first element is the id of the compound,
# the second is the location of the pdb file of the corresponding compound


def read_dataset(dataset):

    if dataset not in ['PP', 'PLnotrefined', 'PL']:
        raise ValueError('argument must be either PP, PLnotrefined or PL')

    pdb_files = []
    directory = diri + dataset_dic[dataset]

    if dataset == 'PL' or dataset == 'PLrefined':

        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            files = os.listdir(f)
            pdb_id = filename
             
            pdb_files += [(pdb_id, os.path.join(f, files[3]),
                           os.path.join(f, files[1]))]

    if dataset == 'PP':

        for filename in os.listdir(directory):

            pdb_files += [(filename[:4], os.path.join(directory, filename))]

    return pdb_files


aff_dict = get_affinities('PL')


class PDBDataset(torch.utils.data.Dataset):
    
    def __init__(self, pdb_files):
        
        """
        Args:
            pdb_files: list with triplets containing 
            name of compound (4 letters)
            path to pdb file describing protein
            path to sdf file describing ligand
        """
        
        self.dataset_len = len(pdb_files)
        
        parser = PDBParser(QUIET=True)
        
        max_len_p = 0
        max_len_l = 0
        
        data = []
        
        for comp_name, path_protein, path_ligand in pdb_files:
            
            structure_pro = parser.get_structure(comp_name, path_protein)
            coord_p = [atom.get_coord() for atom in structure_pro.get_atoms()]
            
            max_len_p = max(max_len_p, len(coord_p))
            
            structure_lig = Chem.SDMolSupplier(path_ligand,sanitize=False)[0]
            conf_lig = structure_lig.GetConformer()
            coord_l = conf_lig.GetPositions()
            
            max_len_l = max(max_len_l, len(coord_l))
            
            data += [(torch.tensor(coord_p), torch.tensor(coord_l),
                     aff_dict[comp_name][2])]
        
        self.data = data
        self.max_len_p = max_len_p
        self.max_len_l = max_len_l
        
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        coords_p, coords_l, affinity = self.data[index]
        
        coords_p = torch.nn.functional.pad(
            coords_p,(0,0,0, self.max_len_p - coords_p.shape[0]), 
            mode = 'constant', value = None)
        
        coords_l = torch.nn.functional.pad(
            coords_l,(0,0,0, self.max_len_l - coords_l.shape[0]), 
            mode = 'constant', value = None)
        
        return torch.flatten(
            torch.cat((coords_p,coords_l), dim = 0).float()), \
            np.float32(affinity)

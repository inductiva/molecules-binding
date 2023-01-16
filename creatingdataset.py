# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:45:27 2023

@author: anaso
"""

import os

dataset_dic = {'PL': 'datasets/v2020-other-PL',
               'PP': 'datasets/PP',
               'PLrefined': 'datasets/refined-set'}

# creates a list, where the first element is the id of the compound, 
# the second is the location of the pdb file of the corresponding compound

def read_dataset(dataset):
    
    if dataset not in ['PP','PL','PLrefined']:
        raise ValueError('argument must be either PP, PL or PLrefined')
    
    pdb_files = [] 
    directory = dataset_dic[dataset]
    
    if dataset == 'PL' or dataset == 'PLrefined':
        
        for filename in os.listdir(directory):
            #print(filename)
            f = os.path.join(directory,filename)
            files = os.listdir(f)
            pdb_id = files[3][:4]
            pdb_files += [(pdb_id, os.path.join(f, files[3]))]
            
    if dataset == 'PP':
        
        for filename in os.listdir(directory):
    
            pdb_files += [(filename[:4], os.path.join(directory, filename))]
        
    return pdb_files


    

    
    
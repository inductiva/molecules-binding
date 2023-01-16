# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:17:14 2023

@author: anaso
"""
import re
import numpy as np

unity_conv = {'mM': -3, 'uM': -6, 'nM': -9, 'pM': -12,'fM': -15}

dataset_dic = {'PL': 'datasets/index/INDEX_general_PL_data.2020',
               'PP': 'datasets/index/INDEX_general_PP.2020'}

def get_affinities(dataset):
    if dataset not in ['PP','PL']:
        raise ValueError('argument must be either PP or PL')
    
    aff_dict = {}
    with open(dataset_dic[dataset], 'r') as f:
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
        
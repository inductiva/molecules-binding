# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:33:28 2023

@author: anaso
"""
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import numpy as np
import torch
from rdkit import Chem

from datasets import get_affinities

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.atomic import add_atomic_edges

from datasets import read_dataset
from torch import nn

from modelGraph import GCN

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

        config = ProteinGraphConfig()
        params_to_change = {
            "granularity": "atom",
            "edge_construction_functions": [add_atomic_edges]
        }
        config = ProteinGraphConfig(**params_to_change)

        for comp_name, path_protein, path_ligand in pdb_files:

            # Ligand
            structure_lig = Chem.SDMolSupplier(path_ligand, sanitize=False)[0]
            conf_lig = structure_lig.GetConformer()
            coords_ligand = torch.as_tensor(conf_lig.GetPositions())

            atoms_ligand_e = [
                atom.GetSymbol() for atom in structure_lig.GetAtoms()
            ]
            num_atoms_ligand = len(atoms_ligand_e)
            atoms_ligand_n = [ele2num[atomtype] for atomtype in atoms_ligand_e]
            atoms_ligand = np.zeros((num_atoms_ligand, num_features))

            atoms_ligand = np.zeros((num_atoms_ligand, num_features))
            atoms_ligand[np.arange(num_atoms_ligand), atoms_ligand_n] = 1
            atoms_ligand = torch.as_tensor(atoms_ligand)

            edges_directed = [[bond.GetBeginAtomIdx(),
                               bond.GetEndAtomIdx()]
                              for bond in structure_lig.GetBonds()]
            edges_bi = []
            for edge in edges_directed:
                i, j = edge
                edges_bi += [[i, j], [j, i]]
            rows_l = [edge[0] for edge in edges_bi]
            cols_l = [edge[1] for edge in edges_bi]
            edges_ligand = torch.tensor([rows_l, cols_l])

            # Protein
            g = construct_graph(config=config, pdb_path=path_protein)
            nodes_dic = {}

            for i, (ident, d) in enumerate(g.nodes(data=True)):
                nodes_dic[ident] = [
                    i + num_atoms_ligand, d["element_symbol"], d["coords"]
                ]

            rows_p = []
            cols_p = []
            edge_attr_p = []
            for u, v, a in g.edges(data=True):
                id1 = nodes_dic[u][0]
                id2 = nodes_dic[v][0]

                rows_p += [id1, id2]
                cols_p += [id2, id1]

                edge_attr_p += [[a["bond_length"], a["kind"]],
                                [a["bond_length"], a["kind"]]]

            edges_protein = torch.as_tensor([rows_p, cols_p])
            nodes_list = list(nodes_dic.values())
            coords_protein = torch.as_tensor(
                [attr[2].tolist() for attr in nodes_list])

            atoms_protein_e = [attr[1] for attr in nodes_list]
            atoms_protein_n = [
                ele2num[atomtype] for atomtype in atoms_protein_e
            ]

            atoms_protein = np.zeros((len(atoms_protein_n), num_features))
            atoms_protein[np.arange(len(atoms_protein_n)), atoms_protein_n] = 1
            atoms_protein = torch.as_tensor(atoms_protein)

            # Graph Protein + Ligand

            atoms = torch.cat((atoms_ligand, atoms_protein))
            coords = torch.cat((coords_ligand, coords_protein))
            edges = torch.cat((edges_ligand, edges_protein), dim=1)

            data_list += [(Data(x=atoms, edge_index=edges,
                                pos=coords), aff_dict[comp_name][2])]

        self.data_list = data_list

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        graph, affinity = self.data_list[index]

        return [graph, torch.as_tensor(float(affinity))]


# Create the dataset object
pdb_files = read_dataset(directory)
dataset = GraphDataset(pdb_files[:5], mydir_aff)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)



input_dim = num_features
hidden_dim = 15
output_dim = 1
mse_loss = nn.MSELoss()
num_epochs = 70
loss_values = []

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim, hidden_dim)# .to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.double().train()
for epoch in range(5):
    epoch_loss = 0
    for inputs, targets in data_loader:
        print(inputs, targets, torch.unsqueeze(targets,-1))
        print(type(inputs), type(torch.unsqueeze(targets,-1)))
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = mse_loss(outputs,  torch.unsqueeze(targets,-1))
        
        loss.backward()
        optimizer.step()
    loss_values.append(epoch_loss / len(data_loader))

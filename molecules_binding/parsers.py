"""
Parsers
"""
from rdkit import Chem
import numpy as np
import torch
import re
import os
from loguru import logger
import warnings
import logging

logger.disable("graphein")
warnings.filterwarnings("ignore")
logging.disable()
# pylint: disable=C0413
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.atomic import add_atomic_edges


def get_affinities(dir_affinity):
    aff_dict = {}
    with open(dir_affinity, "r", encoding="utf-8") as f:
        for line in f:
            if line[0] != "#":
                fields = line.split()
                pdb_id = fields[0]
                log_aff = float(fields[3])
                aff_str = fields[4]
                aff_tokens = re.split("[=<>~]+", aff_str)
                assert len(aff_tokens) == 2
                label, aff_unity = aff_tokens
                assert label in ["Kd", "Ki", "IC50"]
                affinity_value = float(aff_unity[:-2])
                aff = float(affinity_value)
                aff_dict[pdb_id] = [label, aff, log_aff]
    return aff_dict


def read_dataset(directory, which_dataset):
    # creates a list of pdb_id, path to protein, path to ligand
    pdb_files = []
    if which_dataset == "refined_set":
        index = 2
    elif which_dataset == "core_set":
        index = 3

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        files = os.listdir(f)
        pdb_id = filename
        pdb_files += [(pdb_id, os.path.join(f, files[index]),
                       os.path.join(f, files[1]))]

    return pdb_files


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

num_feat = len(ele2num) + 1
num_features = num_feat + 3


def vector2onehot(vector, n_features):
    onehotvector = np.zeros((len(vector), n_features))
    onehotvector[np.arange(len(vector)), vector] = 1
    return onehotvector


config = ProteinGraphConfig()
params_to_change = {
    "granularity": "atom",
    "edge_construction_functions": [add_atomic_edges],
    "verbose": True
}
config = ProteinGraphConfig(**params_to_change)


def ligand_info(path_ligand):
    """
    Parameters
    ----------
    path_ligand : string
        path to the ligand .sdf file.

    Returns
    -------
    ligand_coord : numpy.ndarray
        array of coordinates of ligand.
    atoms_ligand : torch.Tensor
        tensor with one-hot representations of each atom.
    edges_ligand : torch.Tensor
        tensor of the edges between nodes in ligand.
    edges_length_ligand : torch.Tensor
        tensor containing edge lengths.
    num_atoms_ligand : int
        number of atoms of ligand molecule.

    """
    structure_lig = Chem.SDMolSupplier(path_ligand, sanitize=False)[0]
    conf_lig = structure_lig.GetConformer()
    # coordinates of ligand
    ligand_coord = conf_lig.GetPositions()

    # list of element symbols (eg. "H", "O")
    ligand_elems = [atom.GetSymbol() for atom in structure_lig.GetAtoms()]
    num_atoms_ligand = len(ligand_elems)
    # list of corresponding number for each symbol (eg. "H" -> 1)
    ligand_elems_num = [ele2num[atomtype] for atomtype in ligand_elems]

    atoms_ligand = vector2onehot(ligand_elems_num, num_feat)
    atoms_ligand[np.arange(num_atoms_ligand), 0] = 1
    atoms_ligand = torch.as_tensor(atoms_ligand)

    rows_l = []
    cols_l = []

    edges_length_ligand = []
    for bond in structure_lig.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        rows_l += [i, j]
        cols_l += [j, i]
        length = np.linalg.norm(ligand_coord[i] - ligand_coord[j])
        edges_length_ligand += [length, length]

    edges_ligand = torch.as_tensor([rows_l, cols_l])
    edges_length_ligand = torch.as_tensor(edges_length_ligand)

    return (ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand,
            num_atoms_ligand)


def protein_info(path_protein, num_atoms_ligand):
    """
    Parameters
    ----------
    path_protein : string
        path to the protein .pdb file.
    num_atoms_ligand: int
        number of atoms of ligand molecule

    Returns
    -------
    protein_coord : numpy.ndarray
        array of coordinates of protein.
    atoms_protein : torch.Tensor
        tensor with one-hot representations of each atom.
    edges_protein : torch.Tensor
        tensor of the edges between nodes in protein.
    edges_length_protein : torch.Tensor
        tensor containing edge lengths.
    num_atoms_protein : int
        number of atoms of protein molecule.

    """
    g = construct_graph(config=config, pdb_path=path_protein)

    # dictionary for each node returns id number, element symbol, coordinates
    # eg. 'B:LEU:165:CD1': [1073, 'C', array([47.367,  3.943, 37.864])]
    nodes_dict = {}
    for i, (ident, d) in enumerate(g.nodes(data=True)):
        nodes_dict[ident] = [
            i + num_atoms_ligand, d["element_symbol"], d["coords"]
        ]

    num_atoms_protein = len(nodes_dict)

    rows_p = []
    cols_p = []
    edges_length_protein = []
    for u, v, attr in g.edges(data=True):
        id1 = nodes_dict[u][0]
        id2 = nodes_dict[v][0]

        rows_p += [id1, id2]
        cols_p += [id2, id1]
        edges_length_protein += [attr["bond_length"], attr["bond_length"]]

    edges_protein = torch.as_tensor([rows_p, cols_p])
    edges_length_protein = torch.as_tensor(edges_length_protein)

    protein_coord = []
    protein_elems_num = []
    for _, elem, coord in nodes_dict.values():
        protein_coord += [coord.tolist()]
        protein_elems_num += [ele2num[elem]]
    atoms_protein = vector2onehot(protein_elems_num, num_feat)
    atoms_protein = torch.as_tensor(atoms_protein)

    return (protein_coord, atoms_protein, edges_protein, edges_length_protein,
            num_atoms_protein)

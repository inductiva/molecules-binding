"""
Parsers
"""
from rdkit import Chem
import numpy as np
import torch
import re
import os


def get_affinities(affinity_directory):
    affinity_dict = {}
    with open(affinity_directory, "r", encoding="utf-8") as f:
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
                affinity_dict[pdb_id] = [label, aff, log_aff]
    return affinity_dict


def read_dataset(directory, which_dataset, which_file_ligand):
    # creates a list of pdb_id, path to protein, path to ligand
    pdb_files = []
    which_protein_file = {"refined_set": 2, "core_set": 3}
    which_ligand_file = {"sdf": 1, "mol2": 0}

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        files = os.listdir(f)
        pdb_id = filename
        pdb_files += [
            (pdb_id, os.path.join(f, files[which_protein_file[which_dataset]]),
             os.path.join(f, files[which_ligand_file[which_file_ligand]]))
        ]

    return pdb_files


ele2num = {
    "H": 1,
    "O": 2,
    "N": 3,
    "C": 4,
    "S": 5,
    "Se": 6,
    "P": 7,
    "F": 8,
    "Cl": 8,
    "Br": 8,
    "I": 8,
    "Mg": 9,
    "Ca": 9,
    "Sr": 9,
    "Na": 9,
    "K": 9,
    "Cs": 9,
    "Mn": 9,
    "Fe": 9,
    "Co": 9,
    "Ni": 9,
    "Cu": 9,
    "Zn": 9,
    "Cd": 9,
    "Hg": 9
}

num_feat = max(ele2num.values()) + 1
num_features = num_feat + 3


def vector2onehot(vector, n_features):
    onehotvector = np.zeros((len(vector), n_features))
    onehotvector[np.arange(len(vector)), vector] = 1
    return onehotvector


def molecule_info(path, type_mol, num_atoms_additional):
    """from path returns the coordinates, atoms and
    bonds of molecule"""

    if type_mol == "Protein":
        molecule = Chem.MolFromPDBFile(path,
                                       flavor=2,
                                       sanitize=False,
                                       removeHs=False)

    elif type_mol == "Ligand":
        if path[-4:] == ".sdf":
            molecule = Chem.SDMolSupplier(path, sanitize=False,
                                          removeHs=False)[0]
        elif path[-4:] == "mol2":
            molecule = Chem.MolFromMol2File(path,
                                            sanitize=False,
                                            removeHs=False)

    elements_idx = []
    conformer = molecule.GetConformer(0)
    atoms = molecule.GetAtoms()
    num_atoms = molecule.GetNumAtoms()
    for idx in range(num_atoms):
        atom_symbol = atoms[idx].GetSymbol()
        elements_idx += [ele2num[atom_symbol]]
        # TODO(sofia): Add more relevant properties

    atoms = vector2onehot(elements_idx, num_feat)

    if type_mol == "Ligand":
        atoms[np.arange(num_atoms), 0] = 1

    atoms = torch.as_tensor(atoms)

    coords = conformer.GetPositions()

    rows_l = []
    cols_l = []

    edges_length = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        rows_l += [i + num_atoms_additional, j + num_atoms_additional]
        cols_l += [j + num_atoms_additional, i + num_atoms_additional]
        length = np.linalg.norm(coords[i] - coords[j])
        edges_length += [length, length]

    edges = torch.as_tensor([rows_l, cols_l])
    edges_length = torch.as_tensor(edges_length)

    return (coords, atoms, edges, edges_length, num_atoms)

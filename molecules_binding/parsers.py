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
    "H": 0,
    "O": 1,
    "N": 2,
    "C": 3,
    "S": 4,
    "Se": 5,
    "P": 6,
    "F": 7,
    "Cl": 7,
    "Br": 7,
    "I": 7,
    "Mg": 8,
    "Ca": 8,
    "Sr": 8,
    "Na": 8,
    "K": 8,
    "Cs": 8,
    "Mn": 8,
    "Fe": 8,
    "Co": 8,
    "Ni": 8,
    "Cu": 8,
    "Zn": 8,
    "Cd": 8,
    "Hg": 8
}

num_atom_types = max(ele2num.values()) + 1

pt = Chem.GetPeriodicTable()


def molecule_info(path, type_mol, num_atoms_ligand):
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

    atom_features = []
    conformer = molecule.GetConformer(0)
    num_atoms = molecule.GetNumAtoms()

    if type_mol == "Ligand":
        first_elem = [1]
    else:
        first_elem = [0]

    for atom in molecule.GetAtoms():
        atom_symbol = atom.GetSymbol()
        onehot_elem = np.zeros(num_atom_types)
        onehot_elem[ele2num[atom_symbol]] = 1

        onehot_total_valence = np.zeros(8)
        onehot_total_valence[atom.GetTotalValence()] = 1

        onehot_explicit_valence = np.zeros(8)
        onehot_explicit_valence[atom.GetExplicitValence()] = 1

        onehot_implicit_valence = np.zeros(4)
        onehot_implicit_valence[atom.GetImplicitValence()] = 1

        van_der_waals_radius = pt.GetRvdw(atom_symbol)
        covalent_radius = pt.GetRcovalent(atom_symbol)

        atom_features += [[
            *first_elem, *onehot_elem, *onehot_total_valence,
            *onehot_explicit_valence, *onehot_implicit_valence, *[
                atom.GetFormalCharge(),
                atom.IsInRing(), van_der_waals_radius, covalent_radius
            ]
        ]]

    atom_features = torch.as_tensor(atom_features)

    coords = conformer.GetPositions()

    rows_l = []
    cols_l = []

    edges_features = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_in_ring = bond.IsInRing()
        rows_l += [i + num_atoms_ligand, j + num_atoms_ligand]
        cols_l += [j + num_atoms_ligand, i + num_atoms_ligand]
        length = np.linalg.norm(coords[i] - coords[j])

        vector_ij = list(coords[j] - coords[i])
        vector_ji = list(coords[i] - coords[j])

        edges_features += [[*vector_ij, length, bond_type, bond_in_ring],
                           [*vector_ji, length, bond_type, bond_in_ring]]

    edges = torch.as_tensor([rows_l, cols_l])

    edges_features = torch.as_tensor(edges_features)

    return (coords, atom_features, edges, edges_features, num_atoms)

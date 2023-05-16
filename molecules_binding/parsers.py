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
                label, aff_and_unity = aff_tokens
                assert label in ["Kd", "Ki", "IC50"]
                affinity_value = float(aff_and_unity[:-2])
                aff_unity = aff_and_unity[-2:]
                aff = float(affinity_value)
                affinity_dict[pdb_id] = [label, log_aff, aff, aff_unity]
    return affinity_dict


def read_dataset(directory, ligand_file_extention, protein_file_extention):
    """
    from directory returns a list of pdb_id, path to protein, path to ligand
    The directory contains compound folders (each has an ID with 4 letters,
    ex. abcd) with 4 files:
    - abcd_protein.pdb
    - abcd_pocket.pdb
    - abcd_ligand.sdf
    - abcd_ligand.mol2
    """
    assert ligand_file_extention in ("sdf", "mol2")
    assert protein_file_extention in ("protein", "pocket", "processed")
    molecules_files = []
    for folder_name in os.listdir(directory):
        if len(folder_name) == 4:
            folder_dir = os.path.join(directory, folder_name)
            files = os.listdir(folder_dir)
            compound_id = folder_name

            for file in files:
                if file.endswith(protein_file_extention + ".pdb"):
                    file_protein = file
                elif file.endswith("ligand." + ligand_file_extention):
                    file_ligand = file

            molecules_files += [(compound_id,
                                 os.path.join(folder_dir, file_protein),
                                 os.path.join(folder_dir, file_ligand))]

    return molecules_files


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
            suplier = Chem.SDMolSupplier(path, sanitize=False,
                                          removeHs=False)
            molecule = next(suplier)
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
        onehot_elem[ele2num.get(atom_symbol, 8)] = 1

        onehot_total_valence = np.zeros(9)
        onehot_total_valence[atom.GetTotalValence()] = 1

        onehot_explicit_valence = np.zeros(9)
        onehot_explicit_valence[atom.GetExplicitValence()] = 1

        onehot_implicit_valence = np.zeros(5)
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

        vector_ij = list((coords[j] - coords[i]) / length)
        vector_ji = list((coords[i] - coords[j]) / length)

        edges_features += [[*vector_ij, length, bond_type, bond_in_ring],
                           [*vector_ji, length, bond_type, bond_in_ring]]

    edges = torch.as_tensor([rows_l, cols_l])

    edges_features = torch.as_tensor(edges_features)

    return (coords, atom_features, edges, edges_features, num_atoms)

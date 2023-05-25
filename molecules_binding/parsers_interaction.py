"""
Parsers (replicating Interaction Net)
"""

from rdkit import Chem
import numpy as np
import torch

ele2num = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "F": 4,
    "P": 5,
    "Cl": 6,
    "Br": 7,
    "I": 8,
    "B": 9,
    "Si": 10,
    "Fe": 11,
    "Zn": 12,
    "Cu": 13,
    "Mn": 14,
    "Mo": 15
}

hybrid2num = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
    Chem.rdchem.HybridizationType.S: 2,
    Chem.rdchem.HybridizationType.SP2D: 1
}

chirality2num = {"R": 0, "S": 1}

bond2num = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
    Chem.rdchem.BondType.UNSPECIFIED: 4,  # not ideal
    Chem.rdchem.BondType.ZERO: 4  #not ideal
}

stereo2num = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3
}

pt = Chem.GetPeriodicTable()


def molecule_info(path, type_mol, num_atoms_ligand):
    """from path returns the coordinates, atoms and
    bonds of molecule"""

    if type_mol == "Protein":
        molecule = Chem.MolFromPDBFile(path,
                                       flavor=2,
                                       sanitize=True,
                                       removeHs=True)

    elif type_mol == "Ligand":

        if path[-4:] == ".sdf":
            suplier = Chem.SDMolSupplier(path, sanitize=True, removeHs=True)
            molecule = next(suplier)
        elif path[-4:] == "mol2":
            molecule = Chem.MolFromMol2File(path, sanitize=True, removeHs=True)
    try:
        conformer = molecule.GetConformer(0)
    except AttributeError:
        return (None, None, None, None, None)
    atom_features = []
    num_atoms = molecule.GetNumAtoms()

    if type_mol == "Ligand":
        first_elem = [1]
    else:
        first_elem = [0]

    for atom in molecule.GetAtoms():
        atom_symbol = atom.GetSymbol()
        onehot_elem = np.zeros(17)
        onehot_elem[ele2num.get(atom_symbol, 16)] = 1

        onehot_atom_degree = np.zeros(6)
        if atom.GetDegree() > 5:
            return (None, None, None, None, None)

        onehot_atom_degree[atom.GetDegree()] = 1

        onehot_hybridization = np.zeros(6)
        onehot_hybridization[hybrid2num[atom.GetHybridization()]] = 1

        onehot_number_of_hs = np.zeros(5)
        onehot_number_of_hs[atom.GetTotalNumHs()] = 1

        onehot_chirality = np.zeros(3)
        if atom.HasProp("_ChiralityPossible") == 1 and atom.HasProp("_CIPCode"):
            onehot_chirality[chirality2num[atom.GetProp("_CIPCode")]] = 1
        else:
            onehot_chirality[2] = 1

        atom_features += [[
            *first_elem, *onehot_elem, *onehot_atom_degree,
            *[atom.GetFormalCharge(),
              atom.GetNumRadicalElectrons()], *onehot_hybridization,
            *[atom.GetIsAromatic()], *onehot_number_of_hs, *onehot_chirality
        ]]

    atom_features = torch.as_tensor(atom_features)

    coords = conformer.GetPositions()

    rows_l = []
    cols_l = []

    edges_features = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        onehot_bond_type = np.zeros(5)
        onehot_bond_type[bond2num[bond_type]] = 1

        bond_is_conjugated = bond.GetIsConjugated()
        bond_in_ring = bond.IsInRing()

        onehot_bond_stereo = np.zeros(5)
        onehot_bond_stereo[stereo2num.get(bond.GetStereo(), 4)] = 1

        rows_l += [i + num_atoms_ligand, j + num_atoms_ligand]
        cols_l += [j + num_atoms_ligand, i + num_atoms_ligand]

        edges_features += [[
            *onehot_bond_type, *[bond_is_conjugated, bond_in_ring],
            *onehot_bond_stereo, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]]
        edges_features += [[
            *onehot_bond_type, *[bond_is_conjugated, bond_in_ring],
            *onehot_bond_stereo, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]]

    edges = torch.as_tensor([rows_l, cols_l])

    edges_features = torch.as_tensor(edges_features)
    return (coords, atom_features, edges, edges_features, num_atoms)
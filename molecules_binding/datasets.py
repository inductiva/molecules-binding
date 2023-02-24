"""
Define class dataset
"""
from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from rdkit import Chem
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.atomic import add_atomic_edges

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

config = ProteinGraphConfig()
params_to_change = {
    "granularity": "atom",
    "edge_construction_functions": [add_atomic_edges],
    "verbose": True
}
config = ProteinGraphConfig(**params_to_change)


def vector2onehot(vector, n_features):
    onehotvector = np.zeros((len(vector), n_features))
    onehotvector[np.arange(len(vector)), vector] = 1
    return onehotvector


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


def create_edges_protein_ligand(num_atoms_ligand, num_atoms_protein,
                                ligand_coord, protein_coord, threshold):
    """
    Builds the edges between protein and ligand molecules, with length
    under a threshold
    """
    edges_dis_both = []
    rows_both = []
    cols_both = []

    for atom_l in range(num_atoms_ligand):
        for atom_p in range(num_atoms_protein):
            posl = ligand_coord[atom_l]
            posp = protein_coord[atom_p]
            dis = np.linalg.norm(posl - posp)
            if dis <= threshold:
                rows_both += [atom_l, num_atoms_ligand + atom_p]
                cols_both += [num_atoms_ligand + atom_p, atom_l]
                edges_dis_both += [dis, dis]

    edges_both = torch.as_tensor([rows_both, cols_both])
    edges_dis_both = torch.as_tensor(edges_dis_both)
    return edges_both, edges_dis_both


class GraphDataset(Dataset):
    """ builds the graph for each complex"""

    def __init__(self, pdb_files, aff_d, threshold):
        """
        Args:
            pdb_files: list with triplets containing
                name of compound (4 letters)
                path to pdb file describing protein
                path to sdf file describing ligand
            aff_dict: dictionary that for each complex returns affinity data
            threshold: maximum length of edge connecting protein and ligand
        """
        super().__init__("GraphDataset")
        self.dataset_len = len(pdb_files)

        data_list = []

        for comp_name, path_protein, path_ligand in pdb_files:

            (ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand,
             num_atoms_ligand) = ligand_info(path_ligand)

            (protein_coord, atoms_protein, edges_protein, edges_length_protein,
             num_atoms_protein) = protein_info(path_protein, num_atoms_ligand)

            edges_both, edges_dis_both = create_edges_protein_ligand(
                num_atoms_ligand, num_atoms_protein, ligand_coord,
                protein_coord, threshold)

            # concatenate ligand and protein info

            atoms = torch.cat((atoms_ligand, atoms_protein))

            coords_ligand = torch.as_tensor(ligand_coord)
            coords_protein = torch.as_tensor(protein_coord)
            coords = torch.cat((coords_ligand, coords_protein))

            edges = torch.cat((edges_ligand, edges_protein, edges_both), dim=1)

            edges_atrr = torch.cat(
                (edges_length_ligand, edges_length_protein, edges_dis_both))

            atoms_coords = torch.cat((atoms, coords), dim=1)

            # Create object graph
            data_list += [
                Data(x=atoms_coords,
                     edge_index=edges,
                     pos=coords,
                     edge_attr=edges_atrr,
                     y=torch.as_tensor(np.float64(aff_d[comp_name][2])))
            ]

        self.data_list = data_list

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        return self.data_list[index]


class VectorDataset(torch.utils.data.Dataset):
    """ constructs a vector with coordinates padded and flatten
    (both the ligand and protein) and one-hot chemical element"""

    def __init__(self, pdb_files, aff_dict):
        """
        Args:
            pdb_files: list with triplets containing
                name of compound (4 letters)
                path to pdb file describing protein
                path to sdf file describing ligand
            aff_dict: dictionary that for each complex returns affinity data
        """

        self.dataset_len = len(pdb_files)

        max_len_p = 0
        max_len_l = 0

        data = []

        for comp_name, path_protein, path_ligand in pdb_files:

            # ------------ Protein -------------
            g = construct_graph(config=config, pdb_path=path_protein)
            nodes_dic = {}
            for i, (ident, d) in enumerate(g.nodes(data=True)):
                nodes_dic[ident] = [i, d["element_symbol"], d["coords"]]

            nodes_list = list(nodes_dic.values())
            coord_p = [attr[2].tolist() for attr in nodes_list]

            max_len_p = max(max_len_p, len(coord_p))

            elem_p = [attr[1] for attr in nodes_list]
            elem_p_n = [ele2num[i] for i in elem_p]

            onehotelem_p = vector2onehot(elem_p_n, num_feat)
            onehotelem_p[np.arange(len(elem_p_n)), 0] = 1

            # ------------ Ligand -------------

            structure_lig = Chem.SDMolSupplier(path_ligand, sanitize=False)[0]
            conf_lig = structure_lig.GetConformer()
            coord_l = conf_lig.GetPositions()

            elem_l = [atom.GetSymbol() for atom in structure_lig.GetAtoms()]

            elem_l_n = [ele2num[i] for i in elem_l]
            onehotelem_l = vector2onehot(elem_l_n, num_feat)

            max_len_l = max(max_len_l, len(coord_l))

            data += [
                (torch.as_tensor(np.concatenate((onehotelem_p, coord_p),
                                                axis=1)),
                 torch.as_tensor(np.concatenate(
                     (onehotelem_l, coord_l), axis=1)), aff_dict[comp_name][2])
            ]

        self.data = data
        self.max_len_p = max_len_p
        self.max_len_l = max_len_l

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        coords_p, coords_l, affinity = self.data[index]

        coords_p = torch.nn.functional.pad(
            coords_p, (0, 0, 0, self.max_len_p - coords_p.shape[0]),
            mode="constant",
            value=None)

        coords_l = torch.nn.functional.pad(
            coords_l, (0, 0, 0, self.max_len_l - coords_l.shape[0]),
            mode="constant",
            value=None)

        return torch.flatten(
            torch.cat((coords_p,coords_l), dim = 0).float()), \
            np.float64(affinity)

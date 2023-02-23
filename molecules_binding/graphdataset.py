"""
define class graphdataset
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


def ligand_info(path_ligand):
    structure_lig = Chem.SDMolSupplier(path_ligand, sanitize=False)[0]
    conf_lig = structure_lig.GetConformer()
    # coordinates of ligand
    pos_l = conf_lig.GetPositions()

    atoms_ligand_e = [atom.GetSymbol() for atom in structure_lig.GetAtoms()]
    num_atoms_ligand = len(atoms_ligand_e)
    atoms_ligand_n = [ele2num[atomtype] for atomtype in atoms_ligand_e]
    atoms_ligand = np.zeros((num_atoms_ligand, num_feat))

    atoms_ligand = np.zeros((num_atoms_ligand, num_feat))
    atoms_ligand[np.arange(num_atoms_ligand), atoms_ligand_n] = 1
    atoms_ligand[np.arange(num_atoms_ligand), 0] = 1
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

    edges_dis_l = []
    for edge in edges_directed:
        dis = np.linalg.norm(pos_l[edge[0]] - pos_l[edge[1]])
        edges_dis_l += [dis, dis]
    edges_dis_lig = torch.as_tensor(edges_dis_l)

    return (pos_l, atoms_ligand, edges_ligand, edges_dis_lig, num_atoms_ligand)


def protein_info(path_protein, num_atoms_ligand):
    g = construct_graph(config=config, pdb_path=path_protein)

    nodes_dic = {}
    for i, (ident, d) in enumerate(g.nodes(data=True)):
        nodes_dic[ident] = [
            i + num_atoms_ligand, d["element_symbol"], d["coords"]
        ]

    rows_p = []
    cols_p = []
    edges_dis_p = []
    for u, v, a in g.edges(data=True):
        id1 = nodes_dic[u][0]
        id2 = nodes_dic[v][0]

        rows_p += [id1, id2]
        cols_p += [id2, id1]

        edges_dis_p += [a["bond_length"], a["bond_length"]]
    edges_dis_pro = torch.as_tensor(edges_dis_p)

    edges_protein = torch.as_tensor([rows_p, cols_p])
    nodes_list = list(nodes_dic.values())

    pos_p = [attr[2].tolist() for attr in nodes_list]

    atoms_protein_e = [attr[1] for attr in nodes_list]
    atoms_protein_n = [ele2num[atomtype] for atomtype in atoms_protein_e]

    atoms_protein = np.zeros((len(atoms_protein_n), num_feat))
    atoms_protein[np.arange(len(atoms_protein_n)), atoms_protein_n] = 1
    atoms_protein = torch.as_tensor(atoms_protein)
    num_atoms_protein = len(nodes_dic)

    return (pos_p, atoms_protein, edges_protein, edges_dis_pro,
            num_atoms_protein)


def create_edges_p_l(num_atoms_ligand, num_atoms_protein, pos_l, pos_p,
                     threshold):
    edges_dis_both = []
    rows_both = []
    cols_both = []

    for atom_l in range(num_atoms_ligand):
        for atom_p in range(num_atoms_protein - num_atoms_ligand):
            posl = pos_l[atom_l]
            posp = pos_p[atom_p]
            dis = np.linalg.norm(posl - posp)
            if dis < threshold:
                rows_both += [atom_l, num_atoms_ligand + atom_p]
                cols_both += [num_atoms_ligand + atom_p, atom_l]
                edges_dis_both += [dis, dis]

    edges_both = torch.as_tensor([rows_both, cols_both])
    edges_dis_both = torch.as_tensor(edges_dis_both)
    return edges_both, edges_dis_both


class GraphDataset(Dataset):
    """
    Args:
        pdb_files: list with triplets containing
        name of compound (4 letters)
        path to pdb file describing protein
        path to sdf file describing ligand
        mydir_aff: to remove later
    """

    def __init__(self, pdb_files, aff_d, threshold):
        super().__init__("GraphDataset")
        self.dataset_len = len(pdb_files)

        data_list = []

        for comp_name, path_protein, path_ligand in pdb_files:

            (pos_l, atoms_ligand, edges_ligand, edges_dis_lig,
             num_atoms_ligand) = ligand_info(path_ligand)

            (pos_p, atoms_protein, edges_protein, edges_dis_pro,
             num_atoms_protein) = protein_info(path_protein, num_atoms_ligand)

            edges_both, edges_dis_both = create_edges_p_l(
                num_atoms_ligand, num_atoms_protein, pos_l, pos_p, threshold)

            # concatenate ligand and protein info

            atoms = torch.cat((atoms_ligand, atoms_protein))

            coords_ligand = torch.as_tensor(pos_l)
            coords_protein = torch.as_tensor(pos_p)
            coords = torch.cat((coords_ligand, coords_protein))

            edges = torch.cat((edges_ligand, edges_protein, edges_both), dim=1)

            edges_atrr = torch.cat(
                (edges_dis_lig, edges_dis_pro, edges_dis_both))

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

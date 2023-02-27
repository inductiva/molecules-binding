"""
Define class dataset
"""
from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from molecules_binding.parsers import ligand_info, protein_info


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

            (ligand_coord, atoms_ligand, _, _,
             num_atoms_ligand) = ligand_info(path_ligand)

            (protein_coord, atoms_protein, _, _,
             num_atoms_protein) = protein_info(path_protein, num_atoms_ligand)

            max_len_l = max(max_len_l, num_atoms_ligand)
            max_len_p = max(max_len_p, num_atoms_protein)

            data += [(torch.cat((torch.as_tensor(ligand_coord), atoms_ligand),
                                dim=1),
                      torch.cat((torch.as_tensor(protein_coord), atoms_protein),
                                dim=1), aff_dict[comp_name][2])]

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

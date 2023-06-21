"""
Define class dataset
"""
from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from molecules_binding.parsers import molecule_info


def create_edges_protein_ligand(num_atoms_ligand, num_atoms_protein,
                                ligand_coord, protein_coord, threshold):
    """
    Builds the edges between protein and ligand molecules, with length
    under a threshold
    """
    rows_both = []
    cols_both = []
    edges_features = []

    for atom_l in range(num_atoms_ligand):
        for atom_p in range(num_atoms_protein):
            posl = ligand_coord[atom_l]
            posp = protein_coord[atom_p]
            distance = np.linalg.norm(posl - posp)
            if distance <= threshold:
                rows_both += [atom_l, num_atoms_ligand + atom_p]
                cols_both += [num_atoms_ligand + atom_p, atom_l]

                vector_ij = list((posl - posp) / distance)
                vector_ji = list((posp - posl) / distance)

                edges_features += [[*vector_ij, distance, 0, 0],
                                   [*vector_ji, distance, 0, 0]]

    edges_both = torch.as_tensor([rows_both, cols_both])
    edges_features = torch.as_tensor(edges_features)
    return edges_both, edges_features


def rotate_coordinates(coordinates, angles):
    angles_rad = np.radians(angles)

    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                           [0, np.sin(angles_rad[0]),
                            np.cos(angles_rad[0])]])

    rotation_y = np.array([[np.cos(angles_rad[1]), 0,
                            np.sin(angles_rad[1])], [0, 1, 0],
                           [-np.sin(angles_rad[1]), 0,
                            np.cos(angles_rad[1])]])

    rotation_z = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                           [np.sin(angles_rad[2]),
                            np.cos(angles_rad[2]), 0], [0, 0, 1]])

    rotated_coordinates = np.matmul(
        rotation_z, np.matmul(rotation_y, np.matmul(rotation_x,
                                                    coordinates.T))).T
    return rotated_coordinates


class GraphDataset(Dataset):
    """ builds the graph for each complex"""

    def __init__(self, pdb_files, threshold):
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

        for _, path_protein, path_ligand, affinity in pdb_files:

            (ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand,
             num_atoms_ligand) = molecule_info(path_ligand, "Ligand", 0)

            (protein_coord, atoms_protein, edges_protein, edges_length_protein,
             num_atoms_protein) = molecule_info(path_protein, "Protein",
                                                num_atoms_ligand)

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
                     y=torch.as_tensor(np.float64(affinity)))
            ]

        self.data_list = data_list

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        return self.data_list[index]

    def rotate_graph(self, index, angles, no_coords) -> None:
        data = self.data_list[index]
        data.pos = rotate_coordinates(data.pos, angles)
        data.edge_attr[:, :3] = rotate_coordinates(data.edge_attr[:, :3],
                                                   angles)
        if not no_coords:
            data.x[:, -3:] = data.pos

    def remove_coords_from_nodes(self, index) -> None:
        data = self.data_list[index]
        data.x = data.x[:, :-3]


class VectorDataset(torch.utils.data.Dataset):
    """ constructs a vector with coordinates padded and flatten
    (both the ligand and protein) and one-hot chemical element"""

    def __init__(self, pdb_files):
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

        for _, path_protein, path_ligand, affinity in pdb_files:

            (ligand_coord, atoms_ligand, _, _,
             num_atoms_ligand) = molecule_info(path_ligand, "Ligand", 0)

            (protein_coord, atoms_protein, _, _,
             num_atoms_protein) = molecule_info(path_protein, "Protein",
                                                num_atoms_ligand)

            max_len_l = max(max_len_l, num_atoms_ligand)
            max_len_p = max(max_len_p, num_atoms_protein)

            data += [(torch.cat((torch.as_tensor(ligand_coord), atoms_ligand),
                                dim=1),
                      torch.cat((torch.as_tensor(protein_coord), atoms_protein),
                                dim=1), affinity)]

        self.data = data
        self.max_len_p = max_len_p
        self.max_len_l = max_len_l

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        protein, ligand, affinity = self.data[index]

        protein = torch.nn.functional.pad(
            protein, (0, 0, 0, self.max_len_p - protein.shape[0]),
            mode="constant",
            value=None)

        ligand = torch.nn.functional.pad(
            ligand, (0, 0, 0, self.max_len_l - ligand.shape[0]),
            mode="constant",
            value=None)

        return torch.flatten(
            torch.cat((protein,ligand), dim = 0).float()), \
            np.float64(affinity)

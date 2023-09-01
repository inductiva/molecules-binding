"""
Define class Dataset (replicating Jiang, Dejun, et al. "Interactiongraphnet:
A novel and efficient deep graph representation learning framework for
accurate protein–ligand interaction predictions." Journal of medicinal
chemistry 64.24 (2021): 18209-18232.)
https://pubs.acs.org/doi/10.1021/acs.jmedchem.1c01830
"""
from torch_geometric import data
from torch.utils import data as data_utils
import numpy as np
import torch
from molecules_binding import parsers_interaction
from numba import njit


@njit
def structural_info(a, b, c):
    """
    Args:
        a, b, c: numpy arrays representing 3D coordinates of 3 points
    Returns:
        angle between the vectors ab and ac
        area of the triangle abc
        distance between a and c
    """

    ab = b - a
    ac = c - a
    ab_norm = np.linalg.norm(ab)
    ac_norm = np.linalg.norm(ac)
    norm_prod = ab_norm * ac_norm

    cosine_angle = np.dot(ab, ac) / norm_prod
    cosine_angle = max(cosine_angle, -1.0)
    angle = np.arccos(cosine_angle)

    area = 0.5 * norm_prod * np.sin(angle)
    return np.degrees(angle), area, ac_norm


def create_edges_protein_ligand(num_atoms_ligand, num_atoms_protein,
                                ligand_coord, protein_coord, threshold):
    """
    Builds the edges between protein and ligand molecules, with length
    under a threshold
    """
    rows_both = []
    cols_both = []
    edges_features = []

    coords = np.concatenate((ligand_coord, protein_coord), axis=0)

    for atom_l in range(num_atoms_ligand):
        for atom_p in range(num_atoms_ligand,
                            num_atoms_protein + num_atoms_ligand):
            posl = coords[atom_l]
            posp = coords[atom_p]
            distance = np.linalg.norm(posl - posp)
            if distance <= threshold:
                rows_both += [atom_l, atom_p]
                cols_both += [atom_p, atom_l]

                edges_features += [[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, distance * 0.1
                ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, distance * 0.1]]

    num_edges = len(rows_both)

    # create geometric features for each edge
    for i in range(num_edges):
        edge_begin = rows_both[i]
        edge_end = cols_both[i]
        # neighbors of the begin node of the edge
        neighbors = [
            cols_both[j]
            for j in range(0, num_edges)
            if rows_both[j] == edge_begin and j != i
        ]

        if len(neighbors) > 0:
            angles = []
            areas = []
            distances = []
            for neighbor in neighbors:
                angle, area, distance = structural_info(coords[edge_begin],
                                                        coords[edge_end],
                                                        coords[neighbor])
                angles.append(angle)
                areas.append(area)
                distances.append(distance)
            # normalizing values to be in the same order of magnitude
            # (done in the original paper, but was adapted to other values)
            angle_info = [
                np.max(angles) * 0.01,
                np.sum(angles) * 0.01,
                np.mean(angles) * 0.01
            ]
            area_info = [np.max(areas), np.sum(areas), np.mean(areas)]
            distance_info = [
                np.max(distances) * 0.1,
                np.sum(distances) * 0.1,
                np.mean(distances) * 0.1
            ]
        else:
            angle_info = [0, 0, 0]
            area_info = [0, 0, 0]
            distance_info = [0, 0, 0]

        edge_ind_attr = angle_info + area_info + distance_info

        edges_features[i] += edge_ind_attr

    edges_both = torch.as_tensor([rows_both, cols_both], dtype=torch.int64)
    edges_features = torch.as_tensor(edges_features, dtype=torch.float32)

    return edges_both, edges_features


class GraphDataset(data.Dataset):
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

        data_list = []
        correctly_parsed = set()
        not_correctly_parsed = set()
        for (i, (pdb_id, path_protein, path_ligand,
                 affinity)) in enumerate(pdb_files):
            print(i, pdb_id)
            (ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand,
             num_atoms_ligand) = parsers_interaction.molecule_info(
                 path_ligand, "Ligand", 0)

            if ligand_coord is None:
                not_correctly_parsed.add(pdb_id)

            else:
                (protein_coord, atoms_protein, edges_protein,
                 edges_length_protein,
                 num_atoms_protein) = parsers_interaction.molecule_info(
                     path_protein, "Protein", num_atoms_ligand)

                if protein_coord is None:
                    not_correctly_parsed.add(pdb_id)

                else:
                    correctly_parsed.add(pdb_id)

                    # Two types of edges:
                    # edges representing chemical bonds, and the features of
                    # those edges are [chemical features(...), 0,0,...,0]
                    # edges joining the ligand molecule with the protein
                    # molecule, and those edges have features
                    # [0,..,0, 3-dimensional features]

                    edges_both, edges_dis_both = create_edges_protein_ligand(
                        num_atoms_ligand, num_atoms_protein, ligand_coord,
                        protein_coord, threshold)

                    # concatenate ligand and protein info

                    atoms = torch.cat((atoms_ligand, atoms_protein))

                    coords_ligand = torch.as_tensor(ligand_coord)
                    coords_protein = torch.as_tensor(protein_coord)
                    coords = torch.cat((coords_ligand, coords_protein))

                    edges = torch.cat((edges_ligand, edges_protein, edges_both),
                                      dim=1)

                    edges_atrr = torch.cat(
                        (edges_length_ligand, edges_length_protein,
                         edges_dis_both))

                    # Create object graph
                    data_list += [
                        data.Data(x=atoms,
                                  edge_index=edges,
                                  pos=coords,
                                  edge_attr=edges_atrr,
                                  y=[
                                      torch.as_tensor(affinity,
                                                      dtype=torch.float32),
                                      str(pdb_id)
                                  ])
                    ]

        print(correctly_parsed)
        print(not_correctly_parsed)
        print(f"Parsed {len(correctly_parsed)} complexes")
        print(f"Not parsed {len(not_correctly_parsed)} complexes")
        self.data_list = data_list
        self.dataset_len = len(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def shuffle_nodes(self, index) -> None:
        graph = self.data_list[index]
        shuffled_indexes = torch.randperm(graph.x.size(0))
        shuffled_indexes_index = torch.argsort(shuffled_indexes)
        graph.x = graph.x[shuffled_indexes_index]
        graph.pos = graph.pos[shuffled_indexes_index]
        # the edge indexes have to be updated in the same way
        graph.edge_index = shuffled_indexes[graph.edge_index]

    def translate_coords(self, index, translation) -> None:
        graph = self.data_list[index]
        graph.pos += translation

    def remove_graph(self, index):
        del self.data_list[index]

    def remove_graph_by_ids(self, pdb_ids):
        """eliminate complexes based on a list of their ids"""
        indexes_to_del = []
        for i, graph in enumerate(self.data_list):
            if graph.y[1] in pdb_ids:
                indexes_to_del.append(i)
        for index in sorted(indexes_to_del, reverse=True):
            del self.data_list[index]


class VectorDataset(data_utils.Dataset):
    """ constructs a vector with coordinates padded and flatten
    (both the ligand and protein) and one-hot chemical element"""

    def __init__(self, pdb_files, with_coords: bool):
        """
        Args:
            pdb_files: list with triplets containing
                name of compound (4 letters)
                path to pdb file describing protein
                path to sdf file describing ligand
            aff_dict: dictionary that for each complex returns affinity data
        """

        max_len_p = 0
        max_len_l = 0

        data_list = []
        not_correctly_parsed = set()
        correctly_parsed = set()

        for (i, (pdb_id, path_protein, path_ligand,
                 affinity)) in enumerate(pdb_files):
            print(i, pdb_id)

            (ligand_coord, atoms_ligand, _, _,
             num_atoms_ligand) = parsers_interaction.molecule_info(
                 path_ligand, "Ligand", 0)

            if ligand_coord is None:
                not_correctly_parsed.add(pdb_id)

            else:

                (protein_coord, atoms_protein, _, _,
                 num_atoms_protein) = parsers_interaction.molecule_info(
                     path_protein, "Protein", num_atoms_ligand)

                if protein_coord is None:
                    not_correctly_parsed.add(pdb_id)

                else:
                    correctly_parsed.add(pdb_id)
                    max_len_l = max(max_len_l, num_atoms_ligand)
                    max_len_p = max(max_len_p, num_atoms_protein)

                    if with_coords:
                        data_list += [[
                            torch.cat([
                                torch.as_tensor(ligand_coord * 0.1),
                                atoms_ligand
                            ],
                                      dim=1),
                            torch.cat([
                                torch.as_tensor(protein_coord * 0.1),
                                atoms_protein
                            ],
                                      dim=1), affinity
                        ]]
                    else:
                        data_list += [[atoms_ligand, atoms_protein, affinity]]

        self.dataset_len = len(data_list)
        print(correctly_parsed)
        print(not_correctly_parsed)
        print(f"Parsed {len(correctly_parsed)} complexes")
        print(f"Not parsed {len(not_correctly_parsed)} complexes")
        self.data_list = data_list
        self.max_len_p = max_len_p
        self.max_len_l = max_len_l

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        protein, ligand, affinity = self.data_list[index]

        protein = torch.nn.functional.pad(
            protein, (0, 0, 0, self.max_len_p - protein.shape[0]),
            mode="constant",
            value=None)

        ligand = torch.nn.functional.pad(
            ligand, (0, 0, 0, self.max_len_l - ligand.shape[0]),
            mode="constant",
            value=None)

        return [
            torch.flatten(torch.cat((protein, ligand), dim=0).float()),
            np.float64(affinity)
        ]

    def shuffle_nodes(self, index: int) -> None:
        protein, ligand, affinity = self.data_list[index]
        protein = protein[torch.randperm(protein.shape[0])]
        ligand = ligand[torch.randperm(ligand.shape[0])]
        self.data_list[index] = [protein, ligand, affinity]

    def translate_complex(self, index: int) -> None:
        protein, ligand, affinity = self.data_list[index]
        protein[:, -3:] = protein[:, -3:] + 3
        ligand[:, -3:] = ligand[:, -3:] + 3
        self.data_list[index] = [protein, ligand, affinity]

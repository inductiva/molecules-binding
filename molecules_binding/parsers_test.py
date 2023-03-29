""" test parsers functions on an element of the dataset"""
from molecules_binding.parsers import get_affinities
import pytest

path_example_index = "example_dataset/index_data_example.2020"


@pytest.mark.parametrize("affinity_dict, expected_dict",
                         [(get_affinities(path_example_index), {
                             "3zzf": ["Ki", 0.4, 400.0, "mM"],
                             "1hvl": ["Ki", 9.95, 112.0, "pM"],
                             "1zsb": ["Kd", 0.6, 250.0, "mM"],
                             "4ux4": ["IC50", 7.01, 97.0, "nM"],
                             "5a3s": ["Kd", 8.68, 2.1, "nM"],
                             "2uyw": ["Kd", 13.0, 0.1, "pM"]
                         })])
def test_get_affinities(affinity_dict, expected_dict):
    assert affinity_dict == expected_dict


# TODO (Sofia): test graphs creation
# path_dataset = "example_dataset/"

# pdb_files = read_dataset(path_dataset, "sdf")
# comp_name, path_protein, path_ligand = pdb_files[0]

# (ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand,
#  num_atoms_ligand) = molecule_info(path_ligand, "Ligand", 0)
# (protein_coord, atoms_protein, edges_protein, edges_length_protein,
#  num_atoms_protein) = molecule_info(path_protein, "Protein", num_atoms_ligand)

# threshold = 6
# edges_both, edges_dis_both = create_edges_protein_ligand(
# num_atoms_ligand, num_atoms_protein, ligand_coord, protein_coord, threshold)

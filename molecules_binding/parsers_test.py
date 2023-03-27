""" test parsers functions on an element of the dataset"""
from torch import threshold
from molecules_binding.parsers import molecule_info, read_dataset
from molecules_binding.datasets import create_edges_protein_ligand

path_dataset = "example_dataset/"

pdb_files = read_dataset(path_dataset, "sdf")
comp_name, path_protein, path_ligand = pdb_files[0]

(ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand,
 num_atoms_ligand) = molecule_info(path_ligand, "Ligand", 0)
(protein_coord, atoms_protein, edges_protein, edges_length_protein,
 num_atoms_protein) = molecule_info(path_protein, "Protein", num_atoms_ligand)

threshold = 6
edges_both, edges_dis_both = create_edges_protein_ligand(
    num_atoms_ligand, num_atoms_protein, ligand_coord, protein_coord, threshold)

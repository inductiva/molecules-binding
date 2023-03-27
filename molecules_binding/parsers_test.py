from molecules_binding.parsers import molecule_info, read_dataset

path = "../example_dataset/"

pdb_files = read_dataset(path, "refined_set", "sdf")
comp_name, path_protein, path_ligand = pdb_files[0]

(ligand_coord, atoms_ligand, edges_ligand, edges_length_ligand, num_atoms_ligand) = molecule_info(path_ligand, "Ligand", 0)
(protein_coord, atoms_protein, edges_protein, edges_length_protein, num_atoms_protein) = molecule_info(path_protein, "Protein", num_atoms_ligand)
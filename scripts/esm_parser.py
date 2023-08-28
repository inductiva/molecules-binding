import pathlib
import torch
from absl import app
from absl import flags
import esm
from esm import pretrained, MSATransformer, FastaBatchedDataset
from tqdm import tqdm
from Bio import PDB
from typing import List
from rdkit import Chem
from molecules_binding.parsers import get_affinities, CASF_2016_core_set
import os
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "specify the path to the dataset")
flags.DEFINE_string("affinity_dir", None, "specify the path to the index of the dataset")


def read_dataset2(directory, ligand_file_extention, protein_file_extention,
                 aff_dict):
    '''
    from directory returns a list of pdb_id, path to protein, path to ligand
    The directory contains compound folders (each has an ID with 4 letters,
    ex. abcd) with 4 files:
    - abcd_protein.pdb
    - abcd_pocket.pdb
    - abcd_ligand.sdf
    - abcd_ligand.mol2
    '''
    assert ligand_file_extention in ('sdf', 'mol2')
    assert protein_file_extention in ('protein', 'pocket', 'processed')
    molecules_files = []
    for folder_name in os.listdir(directory):
        if len(folder_name) == 4:
            folder_dir = os.path.join(directory, folder_name)
            files = os.listdir(folder_dir)
            compound_id = folder_name

            for file in files:
                if file.endswith(protein_file_extention + '.pdb'):
                    file_pocket = file
                elif file.endswith('ligand.' + ligand_file_extention):
                    file_ligand = file
                elif file.endswith('protein.pdb'):
                    file_protein = file
            if aff_dict[compound_id][4]:
                # only add molecule if affinity is not uncertain
                molecules_files += [
                    (compound_id, os.path.join(folder_dir, file_pocket),
                     os.path.join(folder_dir, file_ligand), os.path.join(folder_dir, file_protein), aff_dict[compound_id][1])
                ]
    return molecules_files



def find_correct_element_of_chain(residue_name, residue_id, atom_symbol, atom_coords, structure):
    for model in structure:
        for i, chain in enumerate(model):
            for j,residue in enumerate(chain):
                if residue.get_resname() == residue_name and residue.get_id()[1] == residue_id:
                    for atom in residue:
                        if atom.element == atom_symbol:
                            if (atom.get_coord() - atom_coords < 0.1).all():
                                return i,j
    return None, None

def pdb_to_sequences2(pdb_id, pdb_filepath: str):
    residues_not_parsed = set()
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(pdb_id, pdb_filepath)
    
    for model in structure:
        chains = []
        for i, chain in enumerate(model):
            chain_str = ""
            for j, residue in enumerate(chain):
                try:
                    chain_str += PDB.Polypeptide.protein_letters_3to1[residue.get_resname()]
                except KeyError:
                    chain_str += ""
                    residues_not_parsed.add(residue.get_resname())
                # print(residue.get_resname(), residue.get_id()[1])
            chains.append(chain_str)
    polypeptide_sequences = [
        chain for chain in chains if chain !=""
    ]
    return structure, polypeptide_sequences, residues_not_parsed

files_with_error = {
    '3p3h', '3p3j', '3p44', '3p55', '4i60', '4jfv', '4jfw', '4jhq', '4um9',
    '2vnp', '3vjs', '2z3z', '2zjw', '3bwf', '2rib', '2jdk', '3pup', '3dcq',
    '2c5y', '1ppw', '4avt', '3wax', '3zp9', '3egk', '2fov', '2q2n', '4c4n',
    '1qon', '4ie3', '3cst', '5a3o', '4ixv', '3wc5', '4hmq', '1rdn', '4z7n',
    '4rlp', '2eep', '4gql', '4m3b', '2foy', '2jdu', '1h07', '2a5b', '2wfg',
    '2jdp', '2vr0', '2ci9', '4hxq', '3vjt', '4daw', '3lp1', '2a3w', '1epq',
    '3v9b', '4hww', '3eyd', '2cfd', '2w08', '4hze', '3whw', '1r1h', '3e9b',
    '4fxz', '4ayu', '3lil', '1ai6', '4z7s', '1sl3', '1esz', '3kck', '4z7f',
    '2z97', '2fou', '2nwn', '2jsd', '2aoh', '2cfg', '2boj', '2boi', '1biw',
    '3gpe', '3e6k', '4mma', '2jdy', '1a7x', '2g83', '4avs', '4kw6', '1z3j',
    '4kcx', '2ork', '1mue', '2brh', '2bv4', '4ob1', '3q4c', '1ksn', '2pll',
    '3kqr', '4fil', '4x1r', '3lik', '3qlb', '1hyz', '2jdn', '2ggx', '4ob0',
    '4m3f', '3udn', '4ixu', '4ob2', '2wl4', '2jdm', '1bm6', '3w8o', '4kai',
    '2ria', '2jdh', '2yak', '4kb7', '1q54', '4mm4', '2os9', '2fm5', '2ggu',
    '4abd', '3lp2', '4u0x', '4no1', '4aoc', '1cps', '4mm6', '4dcx', '3lir',
    '3zdv', '1k2v', '4kbi', '3rj7', '4wkv', '4lv1', '1rdl', '3h9f', '1nu1',
    '4z7q', '4wku', '1rdi', '3m1s', '2jbl', '1bcj', '4wkt', '3zju', '3l2y',
    '4wk2', '2fuu', '4iu0', '4l6q', '3zjt', '1f92', '3fxz', '3bho', '3fy0',
    '1qpf', '1rdj', '4ie2', '4i06', '1g7v'
}

files_with_parsing_error = {
    "1qpb"
}

def all_files(path_aff, path):
    aff_dict = get_affinities(path_aff)
    pdb_files = read_dataset2(path, 'mol2', 'pocket', aff_dict)
    pdb_ids_failed_conformer = []
    elements = dict()
    protein_seqs = dict()
    pdb_ids_failed = []
    residues_failed = set()
    for k, (pdb_id, path_pocket, path_ligand, path_protein, aff) in enumerate(pdb_files):
        if pdb_id not in files_with_error and pdb_id not in CASF_2016_core_set and pdb_id not in files_with_parsing_error:
            print(k, pdb_id)
            # pocket molecule
            molecule = Chem.MolFromPDBFile(path_pocket, flavor=2, sanitize=True, removeHs=True)
            try:
                conformer = molecule.GetConformer(0)
            except Exception:
                pdb_ids_failed_conformer.append(pdb_id)
                continue
            coords = conformer.GetPositions()

            # protein molecule
            structure, protein_sequences, residues_not_parsed = pdb_to_sequences2(pdb_id, path_protein)

            protein_seqs[pdb_id] = protein_sequences

            residues_failed.update(residues_not_parsed)

            # associate atoms of pocket to protein residues
            correct_chain_residue = []
            for n, atom in enumerate(molecule.GetAtoms()):
                atom_coord = coords[n]
                atom_symbol = atom.GetSymbol()
                atom_res = atom.GetPDBResidueInfo()
                atom_res_name = atom_res.GetResidueName()
                atom_res_id = atom_res.GetResidueNumber()
                (i, j) = find_correct_element_of_chain(atom_res_name, atom_res_id, atom_symbol, atom_coord, structure)
                correct_chain_residue.append((i, j))
                if i is not None and j is not None:
                    if i < len(protein_sequences):
                        if j < len(protein_sequences[i]):
                            # print(i, j, PDB.Polypeptide.one_to_three(protein_sequences[i][j]), atom_res_name)
                            if PDB.Polypeptide.one_to_three(protein_sequences[i][j]) != atom_res_name:
                                pdb_ids_failed.append(pdb_id)
                                break
            
            elements[pdb_id] = correct_chain_residue
    return elements, pdb_ids_failed, residues_failed, protein_seqs

def main(_):
    elements, pdb_ids_failed, residues_failed, protein_seqs = all_files(FLAGS.affinity_dir, FLAGS.data_dir)
    print(pdb_ids_failed)
    print(residues_failed)
    with open('../elements.pkl', 'wb') as f:
        pickle.dump(elements, f)
    
    with open('protein_seqs.pkl', 'wb') as f:
        pickle.dump(protein_seqs, f)



if __name__ == "__main__":
    app.run(main)
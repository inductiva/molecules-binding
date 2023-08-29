"""create a new dataset that includes ESM embeddings as node features
(for the pocket atoms)"""
import torch
from absl import app
from absl import flags
from Bio import PDB
from rdkit import Chem
from molecules_binding.parsers import get_affinities, CASF_2016_core_set, files_with_error, read_dataset
import pickle
import esm
from esm import FastaBatchedDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "specify the path to the dataset")
flags.DEFINE_string("affinity_dir", None,
                    "specify the path to the index of the dataset")
flags.DEFINE_string("path_ESM_files", None,
                    "specify a path to store auxiliary files")


def find_correct_element_of_chain(residue_name, residue_id, atom_symbol,
                                  atom_coords, structure):
    """Find the correct element of the chain in protein
    that contains the atom of the pocket molecule.
    i represents the chain number, j the element of the chain"""
    for model in structure:
        for i, chain in enumerate(model):
            for j, residue in enumerate(chain):
                if residue.get_resname() == residue_name and residue.get_id(
                )[1] == residue_id:
                    for atom in residue:
                        if atom.element == atom_symbol:
                            if (atom.get_coord() - atom_coords < 0.1).all():
                                return i, j
    return None, None


def pdb_to_sequences(pdb_id, pdb_filepath: str):
    """Extract the polypeptide sequences from a pdb file"""
    residues_not_parsed = set()
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(pdb_id, pdb_filepath)

    for model in structure:
        chains = []
        for chain in model:
            chain_str = ""
            for residue in chain:
                try:
                    chain_str += PDB.Polypeptide.protein_letters_3to1[
                        residue.get_resname()]
                except KeyError:
                    chain_str += ""
                    residues_not_parsed.add(residue.get_resname())
            chains.append(chain_str)
    polypeptide_sequences = [chain for chain in chains if chain != ""]
    return structure, polypeptide_sequences, residues_not_parsed


files_with_parsing_error = {"1qpb"}


def all_files(path_aff, path):
    """For each pdb_id in the dataset, extract the protein sequence
    eg. protein_seqs["3fwv"]=
    ["SKQALKEKELGNDAYKKKDFDTALKHYDKAKELDPTNMTYIVNQAAVYFEKGDYNKCRELC
    EKAIEVGRENREDYRMIAYAYARIGNSYFKEEKYKDAIHFYNKSLAEHRTPKVLKKCQQAEKILKEQ"]
    each element on the list represents a chain,
    and each letter represents an amino acid.
    For this example, it is a protein with only one chain
    and 128 amino acids.

    elements["3fwv"]=[(0, 4),(0, 4),(0, 4),(0, 4),(0, 4),(0, 4),(0, 4),
    (0, 4),(0, 7),(0, 7),(0, 7),(0, 7),(0, 7),(0, 7),(0, 7),(0, 7),(0, 7),
    (0, 8),(0, 8),(0, 8),...] is a list of tuples for each atom, meaning
    that atom 0 of pocket is (0, 4) from the 4th residue of the 0th chain.

    due to some non-standard amino acids present (residues failed) in the
    pdb files, some complexes could not be correctly parsed.

    """
    aff_dict = get_affinities(path_aff)
    pdb_files = read_dataset(path, "mol2", "pocket", aff_dict, True)
    pdb_ids_failed_conformer = []
    elements = {}
    protein_seqs = {}
    pdb_ids_failed = []
    residues_failed = set()
    for k, (pdb_id, path_pocket, _, path_protein, _) in enumerate(pdb_files):
        if (pdb_id not in files_with_error and
                pdb_id not in CASF_2016_core_set and
                pdb_id not in files_with_parsing_error):
            print(k, pdb_id)
            # pocket molecule
            molecule = Chem.MolFromPDBFile(path_pocket,
                                           flavor=2,
                                           sanitize=True,
                                           removeHs=True)
            try:
                conformer = molecule.GetConformer(0)
            except Exception:
                pdb_ids_failed_conformer.append(pdb_id)
                continue
            coords = conformer.GetPositions()

            # protein molecule
            (structure, protein_sequences,
             residues_not_parsed) = pdb_to_sequences(pdb_id, path_protein)

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
                (i, j) = find_correct_element_of_chain(atom_res_name,
                                                       atom_res_id, atom_symbol,
                                                       atom_coord, structure)
                correct_chain_residue.append((i, j))
                if i is not None and j is not None:
                    if i < len(protein_sequences):
                        if j < len(protein_sequences[i]):
                            if PDB.Polypeptide.one_to_three(
                                    protein_sequences[i][j]) != atom_res_name:
                                pdb_ids_failed.append(pdb_id)
                                break

            elements[pdb_id] = correct_chain_residue
    return elements, pdb_ids_failed, residues_failed, protein_seqs


def get_embeddings_from_sequences(protein_sequences, batch_converter, model,
                                  repr_layers):
    """For each pdb_id the protein sequences are passed to the ESM model"""
    embeddings = {}
    for i, key in enumerate(protein_sequences):
        print(i, key)
        sequences = protein_sequences[key]
        sequences = list(enumerate(sequences))
        batch_labels, batch_strs, _ = batch_converter(sequences)
        dataset = FastaBatchedDataset(batch_labels, batch_strs)
        batches = dataset.get_batch_indices(1, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=batch_converter,
                                                  batch_sampler=batches)

        results = {}
        with torch.no_grad():
            for i, (labels, _, toks) in enumerate(data_loader):
                out = model(toks,
                            repr_layers=repr_layers,
                            return_contacts=False)
                results[labels[0]] = out["representations"][1]

        embeddings[key] = results
    return embeddings


def main(_):
    # 1 - Extract the protein sequences and
    # the correspondences from the pdb files
    elements, pdb_ids_failed, residues_failed, protein_seqs = all_files(
        FLAGS.affinity_dir, FLAGS.data_dir)
    print(pdb_ids_failed)
    print(residues_failed)
    with open(FLAGS.path_esm_files + "/elements.pkl", "wb") as f:
        pickle.dump(elements, f)

    with open(FLAGS.path_esm_files + "./protein_seqs.pkl", "wb") as f:
        pickle.dump(protein_seqs, f)

    with open(FLAGS.path_esm_files + "/pdb_ids_failed.pkl", "wb") as f:
        pickle.dump(pdb_ids_failed, f)

    # 2 - Run the ESM model to extract the embeddings from the protein sequences
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in [8]
    ]

    with open(FLAGS.path_esm_files + "/protein_seqs.pkl", "rb") as f:
        protein_sequences = pickle.load(f)

    embeddings = get_embeddings_from_sequences(protein_sequences,
                                               batch_converter, model,
                                               repr_layers)

    with open(FLAGS.path_esm_files + "/protein_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    # 3 - Include the ESM embeddings in some version of the dataset
    # for files that gave parsing errors, remove them first
    with open(FLAGS.path_esm_files + "/elements.pkl", "rb") as f:
        elements = pickle.load(f)
    with open(FLAGS.path_esm_files + "/protein_embeddings.pkl", "rb") as f:
        protein_embeddings = pickle.load(f)
    with open(FLAGS.path_esm_files + "/pdb_ids_failed.pkl", "wb") as f:
        pickle.dump(pdb_ids_failed, f)

    dataset = torch.load(FLAGS.path_dataset)

    dataset.remove_graph_by_ids(pdb_ids_failed)

    for index, graph in enumerate(dataset):
        print(index, graph.y[1])
        dataset.add_esm_encoding(index, protein_embeddings, elements)

    # save a new dataset with embeddings
    torch.save(dataset, FLAGS.path_dataset_embeddings)


if __name__ == "__main__":
    app.run(main)

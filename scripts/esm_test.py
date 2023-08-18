"""This script extracts embeddings using pre-trained Facebook
ESM models."""
import pathlib
import torch
from absl import app
from absl import flags
from esm import pretrained, MSATransformer, FastaBatchedDataset
from tqdm import tqdm
from Bio import PDB
from typing import List

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_location', 'esm2_t33_650M_UR50D',
    'PyTorch model file OR name of pretrained model to'
    'download (see README for models).')

flags.DEFINE_multi_integer(
    'repr_layers', [33],
    'Layers indices from which to extract representations (0 to'
    'num_layers, inclusive)')

flags.DEFINE_string(
    'input_dir', None,
    'Dataset directory containing the protein sequences to be'
    ' extracted (in the folder PDB). ')

flags.DEFINE_multi_string(
    'embed_mode', ['mean', 'per_tok'],
    'Specify which representations to return, within mean, per_tok,'
    'bos.')

flags.DEFINE_integer('truncation_seq_length', 2000,
                     'Truncate sequences longer than the given value.')

flags.mark_flag_as_required('input_dir')


def pdb_to_sequences(pdb_filepath: str) -> List[str]:
    """
    Reads a PDB file and returns the amino-acid sequence of the protein

    Implemented based on
    https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
    from section:
    'How do I extract polypeptides from a Structure object?'

    Args:
        pdb_filepath: file path to the PDB file
    Returns:
        list of strings containing all polypeptide sequences in the file
    """
    pdb_parser = PDB.PDBParser()
    structure = pdb_parser.get_structure("X", pdb_filepath)
    # Using C-N
    ppb = PDB.PPBuilder()
    polypeptide_sequences = [
        str(pp.get_sequence()) for pp in ppb.build_peptides(structure)
    ]
    return polypeptide_sequences

def parse_pdb_files(folder_path: str):
    """Parse data from folders with the following structure:
    pdb/
    ├─ protein_1.pdb
    ├─ protein_2.pdb

    Where, for example `protein_1.pdb` contains a protein.
    Returns a list with the protein filename and the protein sequence.
    """
    all_pdb_files = list(pathlib.Path(folder_path).glob("*.pdb"))
    data = []
    for file in all_pdb_files:
        protein = pdb_to_sequences(file)[0]
        filename = str(file).replace(".pdb", "")
        data.append((filename, protein))
    return data

def main(_):
    # Load ESM model
    model, alphabet = pretrained.load_model_and_alphabet(FLAGS.model_location)
    batch_converter = alphabet.get_batch_converter(FLAGS.truncation_seq_length)

    model.eval()  # disables dropout for deterministic results

    if isinstance(model, MSATransformer):
        raise ValueError(
            'This script does not handle models with MSA input (MSA'
            'Transformer).')

    if torch.cuda.is_available():
        model = model.cuda()
        print('Transferred model to GPU')

    # Check if output folder exists
    out_dir = pathlib.Path(FLAGS.input_dir).joinpath('esm')
    if not out_dir.exists():
        # Create the folder if it doesn't exist
        out_dir.mkdir(parents=True)
        print(f'Folder created: {out_dir}')
    else:
        print(f'Folder already exists: {out_dir}')

    # Prepare data
    pdb_dir = pathlib.Path(FLAGS.input_dir).joinpath('pdb')
    data = parse_pdb_files(pdb_dir)
    batch_labels, batch_strs, _ = batch_converter(data)
    dataset = FastaBatchedDataset(batch_labels, batch_strs)
    batches = dataset.get_batch_indices(1, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=batch_converter,
                                              batch_sampler=batches)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers
               for i in FLAGS.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1)
                   for i in FLAGS.repr_layers]

    # Extract per-residue representations
    with torch.no_grad():
        for _, (labels, strs, toks) in enumerate(tqdm(data_loader)):
            if torch.cuda.is_available():
                toks = toks.to(device='cuda', non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            _ = out['logits'].to(device='cpu')

            representations = {
                layer: t.to(device='cpu')
                for layer, t in out['representations'].items()
            }
            for i, label in enumerate(labels):
                output_file = out_dir.joinpath(f'{label}.pt')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {'label': label}
                truncate_len = min(FLAGS.truncation_seq_length, len(strs[i]))

                if 'per_tok' in FLAGS.embed_mode:
                    result['per_tok'] = {
                        layer: t[i, 1:truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if 'mean' in FLAGS.embed_mode:
                    result['mean'] = {
                        layer:
                            t[i,
                              1:truncate_len + 1].mean(0).clone().unsqueeze(0)
                        for layer, t in representations.items()
                    }
                if 'bos' in FLAGS.embed_mode:
                    result['bos'] = {
                        layer: t[i, 0].clone()
                        for layer, t in representations.items()
                    }

                torch.save(
                    result,
                    output_file,
                )


if __name__ == '__main__':
    app.run(main)
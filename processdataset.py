"""
Create class dataset from refinedset
"""
import torch
import os
import re
from molecules_binding.datasets import GraphDataset
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("aff_dir", None,
                    "specify the path to the index of the dataset")
flags.mark_flag_as_required("aff_dir")

flags.DEFINE_string("data_dir", None, "specify the path to the dataset")
flags.mark_flag_as_required("data_dir")

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("threshold", 6,
                   "maximum length of edges between protein and ligand")

flags.DEFINE_string("which_dataset", None,
                    "write 'r' if refinedset, 'c' if coreset")
flags.mark_flag_as_required("which_dataset")


def read_dataset(directory, which_dataset):
    # creates a list of pdb_id, path to protein, path to ligand
    pdb_files = []
    assert which_dataset in ("r", "c")
    if which_dataset == "r":
        index = 2
    elif which_dataset == "c":
        index = 3

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        files = os.listdir(f)
        pdb_id = filename
        pdb_files += [(pdb_id, os.path.join(f, files[index]),
                       os.path.join(f, files[1]))]

    return pdb_files


def get_affinities(dir_a):
    # unity_conv = {"mM": -3, "uM": -6, "nM": -9, "pM": -12,"fM": -15}
    aff_dict = {}
    with open(dir_a, "r", encoding="utf-8") as f:
        for line in f:
            if line[0] != "#":
                fields = line.split()
                pdb_id = fields[0]
                log_aff = float(fields[3])
                aff_str = fields[4]
                aff_tokens = re.split("[=<>~]+", aff_str)
                assert len(aff_tokens) == 2
                label, aff_unity = aff_tokens
                assert label in ["Kd", "Ki", "IC50"]
                affinity_value = float(aff_unity[:-2])
                #exponent = unity_conv[aff_unity[-2:]]
                aff = float(affinity_value)
                aff_dict[pdb_id] = [label, aff, log_aff]
    return aff_dict


def create_dataset(direct: str, aff_dir: str, path: str, threshold: float,
                   which_dataset: str):
    pdb_files = read_dataset(direct, which_dataset)
    aff_d = get_affinities(aff_dir)
    datasetg = GraphDataset(pdb_files, aff_d, threshold)
    torch.save(datasetg, path)


def main(_):
    create_dataset(FLAGS.data_dir, FLAGS.aff_dir, FLAGS.path_dataset,
                   FLAGS.threshold, FLAGS.which_dataset)


if __name__ == "__main__":
    app.run(main)

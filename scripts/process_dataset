"""
Create class dataset from refinedset
"""
import torch
from molecules_binding.datasets import GraphDataset
from molecules_binding.parsers import read_dataset, get_affinities
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

flags.DEFINE_enum("which_dataset", "refined_set", ["refined_set", "core_set"],
                  "either refined_set or core_set")
flags.mark_flag_as_required("which_dataset")


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

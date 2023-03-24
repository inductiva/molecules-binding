"""
Create class dataset from refinedset
"""
import torch
from molecules_binding.datasets import GraphDataset
from molecules_binding.datasets import VectorDataset
from molecules_binding.parsers import read_dataset, get_affinities
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("affinity_directory", None,
                    "specify the path to the index of the dataset")
flags.mark_flag_as_required("affinity_directory")

flags.DEFINE_string("data_dir", None, "specify the path to the dataset")
flags.mark_flag_as_required("data_dir")

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("threshold", 6,
                   "maximum length of edges between protein and ligand")

flags.DEFINE_enum("which_dataset", None, ["refined_set", "core_set"],
                  "either refined_set or core_set")
flags.mark_flag_as_required("which_dataset")

flags.DEFINE_enum("which_file_ligand", "sdf", ["sdf", "mol2"],
                  "can choose either mol2 or sdf files")

flags.DEFINE_enum("which_model", "graphnet", ["graphnet", "mlp"],
                  "choose the model")


def create_dataset(direct: str, affinity_directory: str, path: str,
                   threshold: float, which_dataset: str, which_model: str,
                   which_file_ligand: str):
    pdb_files = read_dataset(direct, which_dataset, which_file_ligand)
    affinity_dict = get_affinities(affinity_directory)

    if which_model == "graphnet":
        datasetg = GraphDataset(pdb_files, affinity_dict, threshold)
        torch.save(datasetg, path)
    elif which_model == "mlp":
        datasetv = VectorDataset(pdb_files, affinity_dict)
        torch.save(datasetv, path)


def main(_):
    create_dataset(FLAGS.data_dir, FLAGS.affinity_directory, FLAGS.path_dataset,
                   FLAGS.threshold, FLAGS.which_dataset, FLAGS.which_model,
                   FLAGS.which_file_ligand)


if __name__ == "__main__":
    app.run(main)

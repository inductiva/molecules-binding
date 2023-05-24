"""
Create class dataset from refinedset
"""
import torch
from molecules_binding.datasets_interaction import GraphDataset
from molecules_binding.parsers import read_dataset, get_affinities, CASF_2016_core_set
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("affinity_dir", None,
                    "specify the path to the index of the dataset")
flags.mark_flag_as_required("affinity_dir")

flags.DEFINE_string("data_dir", None, "specify the path to the dataset")
flags.mark_flag_as_required("data_dir")

flags.DEFINE_bool("not_include_test_set", True,
                  "if True, exclude the test set from the dataset")

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("threshold", 6,
                   "maximum length of edges between protein and ligand")

flags.DEFINE_enum("which_file_ligand", "sdf", ["sdf", "mol2"],
                  "can choose either mol2 or sdf files")

flags.DEFINE_enum("which_model", "graphnet", ["graphnet", "mlp"],
                  "choose the model")

flags.DEFINE_enum("which_file_protein", "pocket",
                  ["pocket", "protein", "processed"],
                  "can choose either the entire protein or just the pocket")


def create_dataset(direct: str, affinity_dir: str, path: str, threshold: float,
                   which_model: str, which_file_ligand: str,
                   which_file_protein: str, not_include_test_set: bool):

    affinity_dict = get_affinities(affinity_dir)

    pdb_files = read_dataset(direct, which_file_ligand, which_file_protein,
                             affinity_dict)

    if not_include_test_set:
        pdb_files = [
            pdb_file for pdb_file in pdb_files
            if pdb_file[0] not in CASF_2016_core_set
        ]

    if which_model == "graphnet":
        datasetg = GraphDataset(pdb_files, threshold)
        torch.save(datasetg, path)
    # elif which_model == "mlp":
    #     datasetv = VectorDataset(pdb_files)
    #     torch.save(datasetv, path)


def main(_):
    create_dataset(FLAGS.data_dir, FLAGS.affinity_dir, FLAGS.path_dataset,
                   FLAGS.threshold, FLAGS.which_model, FLAGS.which_file_ligand,
                   FLAGS.which_file_protein, FLAGS.not_include_test_set)


if __name__ == "__main__":
    app.run(main)

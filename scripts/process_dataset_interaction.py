"""
Create class dataset from refinedset
"""
import torch
from molecules_binding import datasets_interaction
from molecules_binding import parsers
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

flags.DEFINE_bool("with_coords_mlp", False,
                  "if True, include coordinates in the mlp dataset")


def create_dataset(direct: str, affinity_dir: str, path: str, threshold: float,
                   which_model: str, which_file_ligand: str,
                   which_file_protein: str, not_include_test_set: bool,
                   with_coords_mlp: bool):

    affinity_dict = parsers.get_affinities(affinity_dir)

    pdb_files = parsers.read_dataset(direct, which_file_ligand,
                                     which_file_protein, affinity_dict)

    if not_include_test_set:
        pdb_files = [
            pdb_file for pdb_file in pdb_files
            if pdb_file[0] not in parsers.CASF_2016_core_set
        ]

    if which_model == "graphnet":
        datasetg = datasets_interaction.GraphDataset(pdb_files, threshold)
        torch.save(datasetg, path)
    elif which_model == "mlp":
        datasetv = datasets_interaction.VectorDataset(pdb_files,
                                                      with_coords_mlp)
        torch.save(datasetv, path)


def main(_):
    create_dataset(FLAGS.data_dir, FLAGS.affinity_dir, FLAGS.path_dataset,
                   FLAGS.threshold, FLAGS.which_model, FLAGS.which_file_ligand,
                   FLAGS.which_file_protein, FLAGS.not_include_test_set,
                   FLAGS.with_coords_mlp)


if __name__ == "__main__":
    app.run(main)

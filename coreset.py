import os
import torch
from molecules_binding.graphdataset import GraphDataset
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("aff_dir",
                    "../../datasets/index/INDEX_general_PL_data.2020",
                    "specify the path to the index of the dataset")

flags.DEFINE_string("data_dir", "../../datasets/CASF-2016/coreset",
                    "specify the path to the dataset")

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("threshold", 6,
                   "maximum length of edges between protein and ligand")


def read_dataset_core(directory):

    pdb_files = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        files = os.listdir(f)
        pdb_id = filename

        pdb_files += [(pdb_id, os.path.join(f, files[3]),
                       os.path.join(f, files[1]))]

    return pdb_files


def main(_):
    pdb_files = read_dataset_core(FLAGS.data_dir)
    datasetcore = GraphDataset(pdb_files, FLAGS.aff_dir, FLAGS.threshold)
    torch.save(datasetcore, FLAGS.path_dataset)


if __name__ == "__main__":
    app.run(main)

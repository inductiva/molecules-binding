# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:37:53 2023

@author: anaso
"""

import torch
from molecules_binding.datasets import read_dataset
from molecules_binding.graphdataset import GraphDataset
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("aff_dir",
                    "../../datasets/index/INDEX_general_PL_data.2020",
                    "specify the path to the index of the dataset")

flags.DEFINE_string("data_dir", "../../datasets/refined-set",
                    "specify the path to the dataset")

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("threshold", 6,
                   "maximum length of edges between protein and ligand")


def create_dataset(direct: str, aff_dir: str, path: str, threshold: float):
    pdb_files = read_dataset(direct)
    datasetg = GraphDataset(pdb_files, aff_dir, threshold)
    torch.save(datasetg, path)


def main(_):
    create_dataset(FLAGS.data_dir, FLAGS.aff_dir, FLAGS.path_dataset,
                   FLAGS.threshold)


if __name__ == "__main__":
    app.run(main)

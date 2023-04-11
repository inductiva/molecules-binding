"""Script to visualize a graph from a dataset."""
import torch
from molecules_binding.visualization_functions import draw_graph
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("path_file", None,
                    "specify the path to an example file in dataset")

flags.DEFINE_string("path_visualization", None,
                    "specify the path to store the image of the visualization")


def main(_):
    graph = torch.load(FLAGS.path_file)
    draw_graph(graph, FLAGS.path_visualization)


if __name__ == "__main__":
    app.run(main)

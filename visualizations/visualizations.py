"""This file will create an image of the created graph for one dataset
complex"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("path_file", None,
                    "specify the path to an example file in dataset")

flags.DEFINE_string(
    "path_visualization", None,
    "specify the path to store the image of the visusalization")


def _format_axes(axe):
    """Visualization options for the 3D axes."""
    axe.grid(False)
    for dim in (axe.xaxis, axe.yaxis, axe.zaxis):
        dim.set_ticks([])
    axe.set_xlabel("x")
    axe.set_ylabel("y")
    axe.set_zlabel("z")


def main(_):
    graph = torch.load(FLAGS.path_file)
    graph_nx = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    positions_dict = {}
    for i in range(graph.pos.size()[0]):
        positions_dict[i] = graph.pos[i].tolist()
    pos = positions_dict

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(graph_nx)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph_nx.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=20, ec="w", c="b")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    _format_axes(ax)
    fig.tight_layout()
    plt.savefig(FLAGS.path_visualization, format="png")
    plt.show()


if __name__ == "__main__":
    app.run(main)

"""This file will create an image of the created graph for one
complex"""
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
import torch
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_integer("element_of_dataset", 0,
                     "specify the element of the dataset to visualize")

flags.DEFINE_string("path_save_figure", "",
                    "directory where to save the visualization")


def _format_axes(axe):
    """Visualization options for the 3D axes."""
    axe.grid(False)
    for dim in (axe.xaxis, axe.yaxis, axe.zaxis):
        dim.set_ticks([])
    axe.set_xlabel("x")
    axe.set_ylabel("y")
    axe.set_zlabel("z")


num_to_contour = {0: "white", 1: "black"}
num_to_color_atoms = {0: "gray", 1: "blue", 2: "red", 3: "cyan", 5: "yellow"}


def draw_graph(graph, path_to_save):
    graph_nx = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    positions_dict = {}
    color_dict = {}
    color_dict_atom = {}
    for i in range(graph.pos.size()[0]):
        positions_dict[i] = graph.pos[i].tolist()
        color_dict[i] = int(graph.x[i][0])
        color_dict_atom[i] = int(graph.x[i][1:10].argmax())
    pos = positions_dict

    node_xyz = np.array([pos[v] for v in sorted(graph_nx)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph_nx.edges()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        *node_xyz.T,
        s=20,
        ec=[num_to_contour[color_dict[i]] for i in range(len(color_dict))],
        c=[
            num_to_color_atoms.get(color_dict_atom[i], "orange")
            for i in range(len(color_dict_atom))
        ],
        linewidths=0.5)

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", linewidth=0.5)

    ax.set_title(f"Complex {graph.y[1]} of dataset")

    _format_axes(ax)
    fig.tight_layout()
    plt.savefig(f"{path_to_save}{graph.y[1]}_visualization.png", format="png")
    plt.show()


def main(_):
    dataset = torch.load(FLAGS.path_dataset)
    graph = dataset[int(FLAGS.element_of_dataset)]
    draw_graph(graph, FLAGS.path_save_figure)


if __name__ == "__main__":
    app.run(main)

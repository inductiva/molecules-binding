"""This file will create an image of the created graph for one dataset
complex"""
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric


def _format_axes(axe):
    """Visualization options for the 3D axes."""
    axe.grid(False)
    for dim in (axe.xaxis, axe.yaxis, axe.zaxis):
        dim.set_ticks([])
    axe.set_xlabel("x")
    axe.set_ylabel("y")
    axe.set_zlabel("z")


num_to_contour = {0: "white", 1: "black"}
num_to_color_atoms = {
    0: "white",
    1: "red",
    2: "blue",
    3: "gray",
    4: "yellow",
    5: "brown",
    6: "orange",
    7: "brown",
    8: "brown"
}


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
            num_to_color_atoms[color_dict_atom[i]]
            for i in range(len(color_dict_atom))
        ],
        linewidths=0.5)

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", linewidth=0.5)

    _format_axes(ax)
    fig.tight_layout()
    plt.savefig(path_to_save, format="png")
    plt.show()

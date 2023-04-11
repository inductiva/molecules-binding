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


def draw_graph(graph, path_to_save):
    graph_nx = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    positions_dict = {}
    for i in range(graph.pos.size()[0]):
        positions_dict[i] = graph.pos[i].tolist()
    pos = positions_dict

    node_xyz = np.array([pos[v] for v in sorted(graph_nx)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph_nx.edges()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*node_xyz.T, s=20, ec="w", c="b")

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    _format_axes(ax)
    fig.tight_layout()
    plt.savefig(path_to_save, format="png")
    plt.show()

"""This file will create an image of the created graph for one dataset
complex"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric

graph = torch.load("visualizations/example_graph")
G = torch_geometric.utils.to_networkx(graph, to_undirected=True)
positions_dict = {}
for i in range(graph.pos.size()[0]):
    positions_dict[i] = graph.pos[i].tolist()
pos = positions_dict

# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=20, ec="w", c="b")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")


def _format_axes(axe):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    axe.grid(False)
    # Suppress tick labels
    for dim in (axe.xaxis, axe.yaxis, axe.zaxis):
        dim.set_ticks([])
    # Set axes labels
    axe.set_xlabel("x")
    axe.set_ylabel("y")
    axe.set_zlabel("z")


_format_axes(ax)
fig.tight_layout()
plt.savefig("visualizations/graph_visualization.png")
plt.show()

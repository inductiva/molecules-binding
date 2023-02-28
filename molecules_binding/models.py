"""
Define models
"""
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    """ Simple Multilayer perceptron """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        layer_sizes = [input_size] + hidden_size
        layers = []
        for ins, outs in list(zip(layer_sizes, layer_sizes[1:])):
            layers.append(nn.Linear(ins, outs))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class GraphNN(MessagePassing):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, hidden_channels, num_node_features):
        """
        Parameters:
            hidden_channels (int): Number of features for each node
            num_node_features (int): Initial number of node features
        """
        super().__init__(aggr='add')

        layer_sizes = [num_node_features] + hidden_channels

        layers = []
        pairs = list(zip(layer_sizes, layer_sizes[1:]))

        for ins, outs in pairs[:-1]:
            layers.append(GATConv(ins, outs))
            layers.append(nn.ReLU())

        for ins, outs in pairs[-1]:
            layers.append(GCNConv(ins, outs))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.lin = nn.Linear(layer_sizes[-1], 1)

    def forward(self, data, batch, dropout_rate):
        """
        Returns:
            x (float): affinity of the graph
        """
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attr
        x = self.layers(x, edge_index, edge_attrs)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.lin(x)

        return x

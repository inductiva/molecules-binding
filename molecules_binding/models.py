"""
Define models
"""
from torch import nn
import torch.nn.functional as F
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

    def __init__(self, num_node_features, layer_sizes_graph,
                 layer_sizes_linear):
        """
        Parameters:
            hidden_channels (int): Number of features for each node
            num_node_features (int): Initial number of node features
        """
        super().__init__(aggr='add')

        graph_layer_sizes = [num_node_features] + layer_sizes_graph
        graph_layers = []
        pairs_graph = list(zip(graph_layer_sizes, graph_layer_sizes[1:]))

        for ins, outs in pairs_graph:
            graph_layers.append(GATConv(ins, outs))

        self.graph_layers = nn.ModuleList(graph_layers)

        linear_layer_sizes = [layer_sizes_graph[-1]] + layer_sizes_linear
        linear_layers = []
        pairs_linear = list(zip(linear_layer_sizes, linear_layer_sizes[1:]))
        for ins, outs in pairs_linear:
            linear_layers.append(nn.Linear(ins, outs))

        self.linear_layers = nn.ModuleList(linear_layers)
        self.last_layer = nn.Linear(linear_layer_sizes[-1], 1)

    def forward(self, data, batch, dropout_rate):
        """
        Returns:
            x (float): affinity of the graph
        """
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attr

        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_attrs)
            x = nn.ReLU()(x)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=dropout_rate, training=self.training)

        for layer in self.linear_layers:
            x = layer(x)
            x = nn.ReLU()(x)
            x = F.dropout(x, p=dropout_rate, training=self.training)

        x = self.last_layer(x)

        return x

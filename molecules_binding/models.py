"""
Define models
"""
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    """ Simple Multilayer perceptron """

    def __init__(self, input_size, hidden_size, output_size, use_batch_norm,
                 dropout_rate):
        super().__init__()

        layer_sizes = [input_size] + hidden_size

        layers = []
        for ins, outs in list(zip(layer_sizes, layer_sizes[1:])):
            layers.append(nn.Linear(ins, outs))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(outs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(layer_sizes[-1], output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class GraphNN(nn.Module):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, num_node_features, layer_sizes_graph, layer_sizes_linear,
                 use_batch_norm, dropout_rate, use_node_embedding,
                 embedding_layers):
        """
        Parameters:
            hidden_channels (int): Number of features for each node
            num_node_features (int): Initial number of node features
        """
        super().__init__()
        graph_layer_sizes = [num_node_features] + layer_sizes_graph
        self.embedding = nn.Identity()

        if use_node_embedding:
            self.embedding = MLP(num_node_features, embedding_layers[:-1],
                                 embedding_layers[-1], use_batch_norm,
                                 dropout_rate)
            graph_layer_sizes[0] = embedding_layers[-1]

        graph_layers = []
        pairs_graph = list(zip(graph_layer_sizes, graph_layer_sizes[1:]))
        batch_norm_layers = []

        for ins, outs in pairs_graph:
            graph_layers.append(GATConv(ins, outs))
            if use_batch_norm:
                batch_norm_layers.append(nn.BatchNorm1d(outs))
            else:
                batch_norm_layers.append(nn.Identity())

        self.graph_layers = nn.ModuleList(graph_layers)
        self.batch_norm_layers = nn.ModuleList(batch_norm_layers)
        self.activation = nn.ReLU()

        self.final_mlp = MLP(layer_sizes_graph[-1], layer_sizes_linear, 1,
                             use_batch_norm, dropout_rate)

    def forward(self, data, batch, dropout_rate, use_message_passing):
        """
        Returns:
            x (float): affinity of the graph
        """
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x)

        if use_message_passing:
            for i, layer in enumerate(self.graph_layers):
                x = layer(x, edge_index, edge_attrs)
                x = self.batch_norm_layers[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=dropout_rate, training=self.training)

        x = self.final_mlp(x)

        return x

"""
Define models
"""
from torch import nn, cat
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric import nn as gnn


class MLP(nn.Module):
    """ Simple Multilayer perceptron """

    def __init__(self, input_size, hidden_size, output_size, use_batch_norm,
                 dropout_rate, final_activation):
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
        if final_activation:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class GraphNN(nn.Module):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, num_node_features, layer_sizes_graph, layer_sizes_linear,
                 use_batch_norm, dropout_rate, embedding_layers,
                 n_attention_heads):
        """
        Parameters:
            hidden_channels (int): Number of features for each node
            num_node_features (int): Initial number of node features
        """
        super().__init__()

        if embedding_layers is None:
            self.embedding = nn.Identity()
            graph_layer_sizes = [num_node_features] + layer_sizes_graph
        else:
            self.embedding = MLP(num_node_features, embedding_layers[:-1],
                                 embedding_layers[-1], use_batch_norm,
                                 dropout_rate, False)
            graph_layer_sizes = [embedding_layers[-1]] + layer_sizes_graph

        graph_layers = []
        pairs_graph = list(zip(graph_layer_sizes, graph_layer_sizes[1:]))
        batch_norm_layers = []

        for ins, outs in pairs_graph:
            graph_layers.append(
                gnn.GATConv(ins, outs, heads=n_attention_heads, concat=False))
            if use_batch_norm:
                batch_norm_layers.append(nn.BatchNorm1d(outs))
            else:
                batch_norm_layers.append(nn.Identity())

        self.graph_layers = nn.ModuleList(graph_layers)
        self.batch_norm_layers = nn.ModuleList(batch_norm_layers)
        self.activation = nn.ReLU()

        self.final_mlp = MLP(layer_sizes_graph[-1], layer_sizes_linear, 1,
                             use_batch_norm, dropout_rate, False)

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

        x = gnn.global_mean_pool(x, batch)
        x = F.dropout(x, p=dropout_rate, training=self.training)

        x = self.final_mlp(x)

        return x


# New approach: Victor's model


class MGNProcessorLayer(MessagePassing):
    """Message passing with edge and node updates"""

    def __init__(self, latent_size):
        super().__init__()

        self.edge_mlp = MLP(3 * latent_size, [latent_size],
                            latent_size,
                            use_batch_norm=True,
                            dropout_rate=0.0,
                            final_activation=True)
        self.node_mlp = MLP(2 * latent_size, [latent_size],
                            latent_size,
                            use_batch_norm=True,
                            dropout_rate=0.0,
                            final_activation=True)

    def forward(self, graph: Batch) -> Batch:

        aggregated_edges, updated_edges = self.propagate(
            edge_index=graph.edge_index, x=graph.x, edge_attr=graph.edge_attr)

        updated_nodes = cat([graph.x, aggregated_edges], dim=1)

        updated_nodes = self.node_mlp(updated_nodes) + graph.x

        graph.x = updated_nodes
        graph.edge_attr = updated_edges

        return graph

    def message(self, x_i, x_j, edge_attr):
        updated_edges = cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        _, target = edge_index

        aggregated_edges = scatter_sum(updated_edges, target, dim=0)

        return aggregated_edges, updated_edges


class MGNProcessor(nn.Module):
    """ Processor for the MGN model"""

    def __init__(self, latent_size, message_passing_steps):
        super().__init__()

        self._latent_size = latent_size
        self._message_passing_steps = message_passing_steps

        self.processor = self._build_processor()

    def _build_processor(self):

        layers = []
        for _ in range(self._message_passing_steps):
            layers.append(MGNProcessorLayer(self._latent_size))

        return nn.Sequential(*layers)

    def forward(self, graph: Batch) -> Batch:
        return self.processor(graph)


class NodeEdgeGNN(nn.Module):
    """ Node and Edge GNN"""

    def __init__(self, num_node_features, num_edge_features, layer_sizes_linear,
                 use_batch_norm, dropout_rate, embedding_layers, latent_size,
                 num_processing_steps):
        super().__init__()

        if embedding_layers is None:
            self.embedding = nn.Identity()
        else:
            self.embedding_nodes = MLP(num_node_features, embedding_layers[:-1],
                                       embedding_layers[-1], use_batch_norm,
                                       dropout_rate, False)
            self.embedding_edges = MLP(num_edge_features, embedding_layers[:-1],
                                       embedding_layers[-1], use_batch_norm,
                                       dropout_rate, False)

        self.processor = MGNProcessor(latent_size, num_processing_steps)

        self.final_mlp = MLP(latent_size * 2, layer_sizes_linear, 1,
                             use_batch_norm, dropout_rate, False)

    def forward(self, data, batch, dropout_rate, use_message_passing):

        data.x = self.embedding_nodes(data.x)
        data.edge_attr = self.embedding_edges(data.edge_attr)

        if use_message_passing:
            data = self.processor(data)

        x = scatter_mean(data.x, batch, dim=0)
        edge = scatter_mean(data.edge_attr, batch[data.edge_index[0]], dim=0)
        aggregation = cat([x, edge], dim=1)
        aggregation = F.dropout(aggregation,
                                p=dropout_rate,
                                training=self.training)

        out = self.final_mlp(aggregation)

        return out

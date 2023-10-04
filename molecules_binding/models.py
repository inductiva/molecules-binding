"""
Define models
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
import torch_scatter


class MLP(nn.Module):
    """ Simple Multilayer perceptron """

    def __init__(self, input_size, hidden_size, output_size, use_batch_norm,
                 dropout_rate, use_final_activation):
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
        if use_final_activation:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class GATProcessor(nn.Module):
    """ Processor for the NodeEdge model"""

    def __init__(self, graph_layer_sizes, use_batch_norm, n_attention_heads):
        super().__init__()

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

    def forward(self, x, edge_index, edge_attr, dropout_rate):
        for norm_layer, layer in zip(self.batch_norm_layers, self.graph_layers):
            x = layer(x, edge_index, edge_attr)
            x = norm_layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=dropout_rate, training=self.training)
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

        self.processor = GATProcessor(graph_layer_sizes, use_batch_norm,
                                      n_attention_heads)

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
            x = self.processor(x, edge_index, edge_attrs, dropout_rate)

        x = gnn.global_mean_pool(x, batch)
        x = F.dropout(x, p=dropout_rate, training=self.training)

        x = self.final_mlp(x)

        return x


class NodeEdgeProcessorLayer(gnn.conv.MessagePassing):
    """Message passing with edge and node updates
    (Message Passing Layer based on
    https://github.com/inductiva/meshnets model)"""

    def __init__(self, latent_size):
        super().__init__()

        self.edge_mlp = MLP(3 * latent_size, [latent_size],
                            latent_size,
                            use_batch_norm=True,
                            dropout_rate=0.0,
                            use_final_activation=True)
        self.node_mlp = MLP(2 * latent_size, [latent_size],
                            latent_size,
                            use_batch_norm=True,
                            dropout_rate=0.0,
                            use_final_activation=True)

    def forward(self, x, edge_index, edge_attr):

        aggregated_edges, updated_edges = self.propagate(edge_index=edge_index,
                                                         x=x,
                                                         edge_attr=edge_attr)

        updated_nodes = torch.cat([x, aggregated_edges], dim=1)

        updated_nodes = self.node_mlp(updated_nodes) + x

        x = updated_nodes
        edge_attr = updated_edges

        return x, edge_attr, edge_index

    # returns the edges after passing through the MLP
    def message(self, x_i, x_j, edge_attr):
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    # aggregates the edges
    def aggregate(self, updated_edges, edge_index, x):

        _, target = edge_index

        aggregated_edges = torch_scatter.scatter_sum(updated_edges,
                                                     target,
                                                     dim=0,
                                                     dim_size=x.size(0))

        return aggregated_edges, updated_edges


class NodeEdgeProcessor(nn.Module):
    """ Processor for the NodeEdge model"""

    def __init__(self, latent_size, message_passing_steps):
        super().__init__()

        self._latent_size = latent_size
        self._message_passing_steps = message_passing_steps

        self.processor = self._build_processor()

    def _build_processor(self):
        layers = []
        for _ in range(self._message_passing_steps):
            layers.append(NodeEdgeProcessorLayer(self._latent_size))

        return nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr):
        for layer in self.processor:
            x, edge_attr, edge_index = layer(x, edge_index, edge_attr)
        return x, edge_index, edge_attr


class NodeEdgeGNN(nn.Module):
    """ Node and Edge GNN"""

    def __init__(self, num_node_features, num_edge_features, layer_sizes_linear,
                 use_batch_norm, dropout_rate, embedding_layers, latent_size,
                 num_processing_steps, final_aggregation, what_to_aggregate):
        super().__init__()

        if embedding_layers is None:
            self.embedding = nn.Identity()
        else:
            self.embedding_nodes = MLP(num_node_features,
                                       embedding_layers[:-1],
                                       embedding_layers[-1],
                                       use_batch_norm,
                                       dropout_rate,
                                       use_final_activation=False)
            self.embedding_edges = MLP(num_edge_features,
                                       embedding_layers[:-1],
                                       embedding_layers[-1],
                                       use_batch_norm,
                                       dropout_rate,
                                       use_final_activation=False)

        self.processor = NodeEdgeProcessor(latent_size, num_processing_steps)

        if what_to_aggregate == 'nodes' or what_to_aggregate == 'edges':
            beggining_layer = latent_size
        elif what_to_aggregate == 'both':
           beggining_layer = latent_size * 2

        self.final_mlp = MLP(beggining_layer,
                             layer_sizes_linear,
                             1,
                             use_batch_norm,
                             dropout_rate,
                             use_final_activation=False)
        self.final_aggregation = final_aggregation
        self.what_to_aggregate = what_to_aggregate

    def forward(self, data, batch, dropout_rate, use_message_passing):

        x = self.embedding_nodes(data.x)
        edge_attr = self.embedding_edges(data.edge_attr)
        edge_index = data.edge_index

        if use_message_passing:
            x, edge_index, edge_attr = self.processor(x, edge_index, edge_attr)

        if self.what_to_aggregate == 'nodes':
            x_mean = torch_scatter.scatter_mean(x, batch, dim=0)
            aggregation = x_mean

        elif self.what_to_aggregate == 'edges':
            edge_mean = torch_scatter.scatter_mean(edge_attr,
                                                   batch[edge_index[0]],
                                                   dim=0)
            aggregation = edge_mean
        elif self.what_to_aggregate == 'both':
            x_mean = torch_scatter.scatter_mean(x, batch, dim=0)
            edge_mean = torch_scatter.scatter_mean(edge_attr,
                                                    batch[edge_index[0]],
                                                    dim=0)
            aggregation = torch.cat([x_mean, edge_mean], dim=1)
        # if self.final_aggregation == 'mean':
        #     x_mean = torch_scatter.scatter_mean(x, batch, dim=0)
        #     edge_mean = torch_scatter.scatter_mean(edge_attr,
        #                                         batch[edge_index[0]],
        #                                         dim=0)
        # elif self.final_aggregation == 'sum':
        #     x_mean = torch_scatter.scatter_sum(x, batch, dim=0)
        #     edge_mean = torch_scatter.scatter_sum(edge_attr,
        #                                         batch[edge_index[0]],
        #                                         dim=0)
        aggregation = F.dropout(aggregation,
                                p=dropout_rate,
                                training=self.training)

        out = self.final_mlp(aggregation)

        return out


class SeparateEdgesGNN(nn.Module):
    """ Node and Edge GNN"""

    def __init__(self, num_node_features, num_edge_features, layer_sizes_linear,
                 use_batch_norm, dropout_rate, embedding_layers, latent_size,
                 num_processing_steps, n_attention_heads, graph_layer_sizes):
        super().__init__()

        if embedding_layers is None:
            self.embedding = nn.Identity()
        else:
            self.embedding_nodes = MLP(num_node_features,
                                       embedding_layers[:-1],
                                       embedding_layers[-1],
                                       use_batch_norm,
                                       dropout_rate,
                                       use_final_activation=False)
            self.embedding_edges = MLP(num_edge_features,
                                       embedding_layers[:-1],
                                       embedding_layers[-1],
                                       use_batch_norm,
                                       dropout_rate,
                                       use_final_activation=False)

        self.processor1 = GATProcessor(graph_layer_sizes, use_batch_norm,
                                       n_attention_heads)

        self.processor2 = NodeEdgeProcessor(latent_size, num_processing_steps)

        self.final_mlp = MLP(latent_size * 2,
                             layer_sizes_linear,
                             1,
                             use_batch_norm,
                             dropout_rate,
                             use_final_activation=False)

    def forward(self, data, batch, dropout_rate, use_message_passing):
        x = self.embedding_nodes(data.x)
        edge_attr_2 = self.embedding_edges(data.edge_attr_2)
        edge_index_2 = data.edge_index_2
        edge_attr_1 = data.edge_attr_1
        edge_index_1 = data.edge_index_1

        if use_message_passing:
            x = self.processor1(x, edge_index_1, edge_attr_1, dropout_rate)

            x, edge_index_2, edge_attr_2 = self.processor2(
                x, edge_index_2, edge_attr_2)

        x_mean = torch_scatter.scatter_mean(x, batch, dim=0)
        edge_mean = torch_scatter.scatter_mean(edge_attr_2,
                                               batch[edge_index_2[0]],
                                               dim=0)
        aggregation = torch.cat([x_mean, edge_mean], dim=1)
        aggregation = F.dropout(aggregation,
                                p=dropout_rate,
                                training=self.training)

        out = self.final_mlp(aggregation)

        return out

import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    """ Simple Multilayer perceptron """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        return x


class GraphNN(MessagePassing):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, hidden_channels, num_node_features):
        """Graph Convolutional Neural Network

        Parameters:
            hidden_channels (int): Number of features for each node
            num_node_features (int): Initial number of node features
        """
        super().__init__(aggr='add')
        torch.manual_seed(12345)
        # self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.gat1 = GATConv(num_node_features, hidden_channels[0])
        self.gat2 = GATConv(hidden_channels[0], hidden_channels[1])
        # self.gat3 = GATConv(hidden_channels[1], hidden_channels[2])
        self.conv1 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv2 = GCNConv(hidden_channels[2], hidden_channels[3])
        # self.lin1 = Linear(hidden_channels[3], hidden_channels[4])
        self.lin1 = Linear(hidden_channels[3], 1)

    def forward(self, data, batch):
        """
        Returns:
            x (float): affinity of the graph
        """
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attr
        # 1. Obtain node embeddings
        x = self.gat1(x, edge_index, edge_attrs)
        x = x.relu()
        x = self.gat2(x, edge_index, edge_attrs)
        x = x.relu()
        # x = self.gat3(x, edge_index, edge_attrs)
        # x = x.relu()
        x = self.conv1(x, edge_index, edge_attrs)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attrs)
        x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)
        # x = x.relu()
        # x = self.lin2(x)

        return x

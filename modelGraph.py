# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:32:55 2023

@author: anaso
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool


class GCN2(MessagePassing):
    def __init__(self, hidden_channels, num_node_features):
        super().__init__(aggr='add')
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# model = GCN2(hidden_channels=64, num_node_features=11)
# model.double()
# print(model)
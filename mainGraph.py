# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:01:43 2023

@author: anaso
"""
from modelGraph import GCN2
from molecules_binding.datasets import read_dataset
from molecules_binding.graphdataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch import nn
import torch

mydir_aff = "../../datasets/index/INDEX_general_PL_data.2020"
directory = "../../datasets/refined-set"

# Create the dataset object
pdb_files = read_dataset(directory)
dataset = GraphDataset(pdb_files[:5], mydir_aff)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

ele2num = {
    "C": 0,
    "H": 1,
    "O": 2,
    "N": 3,
    "S": 4,
    "SE": 5,
    "P": 6,
    "F": 7,
    "Cl": 8,
    "I": 9,
    "Br": 10
}
num_features = len(ele2num)

input_dim = num_features
hidden_dim = 15
output_dim = 1
mse_loss = nn.MSELoss()
num_epochs = 70
loss_values = []

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN2(hidden_channels=64, num_node_features=11)# .to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.double().train()
for epoch in range(5):
    epoch_loss = 0
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs, inputs.batch)
        loss = mse_loss(outputs,  torch.unsqueeze(targets,-1).to(torch.double))
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        
    loss_values.append(epoch_loss / len(data_loader))
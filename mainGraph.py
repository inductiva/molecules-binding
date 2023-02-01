# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:01:43 2023

@author: anaso
"""
from models import GCN
from molecules_binding.datasets import read_dataset
from molecules_binding.graphdataset import GraphDataset
from torch_geometric.loader import DataLoader
import torch

mydir_aff = "../../datasets/index/INDEX_general_PL_data.2020"
directory = "../../datasets/refined-set"

# Create the dataset object
pdb_files = read_dataset(directory)
dataset = GraphDataset(pdb_files, mydir_aff)

train_size = 0.8
train_size = int(train_size * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)

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

model = GCN(hidden_channels=64, num_node_features=num_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


def train():
    model.double().train()

    for data, target in train_loader:
        out = model(data, data.batch)
        loss = criterion(out,
                         torch.unsqueeze(target, -1).to(
                             torch.double))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader):
    model.eval()

    mse = 0
    for data, target in loader:
        out = model(data, data.batch)
        mse += criterion(
            out,
            torch.unsqueeze(target, -1).to(torch.double))
    return mse / len(loader.dataset)


train_errors = []
test_errors = []
for epoch in range(1, 50):
    train()
    train_err = test(train_loader)
    test_err = test(test_loader)
    print(
        f"Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
            Test Error: {test_err:.4f}"
    )
    train_errors += [train_err]
    test_errors += [test_err]
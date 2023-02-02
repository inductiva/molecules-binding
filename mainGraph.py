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
import matplotlib.pyplot as plt

mydir_aff = "../../datasets/index/INDEX_general_PL_data.2020"
directory = "../../datasets/refined-set"


# Create the dataset object
def create_dataset(direct: str, aff_dir: str, path: str):
    pdb_files = read_dataset(direct)
    dataset = GraphDataset(pdb_files, aff_dir)
    torch.save(dataset, path)


if __name__ == "__main__":

    dataset = torch.load("C:/Users/anaso/Desktop/datasetdistance2.pt")
    train_size = 0.8
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_features = 12

    model = GCN(hidden_channels=30, num_node_features=num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    def train():
        model.double().train()
        for data in train_loader:
            out = model(data, data.batch)
            loss = criterion(out, data.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        model.eval()
        mse = 0
        for data in loader:
            out = model(data, data.batch)
            # print(out, data.y)
            mse += criterion(out, data.y.unsqueeze(1))
            print(criterion(out, data.y))
        return mse / len(loader.dataset)

    train_errors = []
    test_errors = []
    for epoch in range(1, 50):
        train()
        train_err = test(train_loader)
        test_err = test(test_loader)
        print(f"Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
                Test Error: {test_err:.4f}")
        train_errors += [train_err]
        test_errors += [test_err]

    plt.plot([elem.detach().numpy() for elem in train_errors])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('train_error.png', bbox_inches='tight')
    plt.show()

    plt.plot([elem.detach().numpy() for elem in test_errors])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('test_error.png', bbox_inches='tight')
    plt.show()

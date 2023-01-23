# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:46:38 2023

@author: anaso
"""
import torch
import torch.nn as nn
from molecules_binding.datasets import read_dataset
from molecules_binding.datasets import get_affinities
from molecules_binding.datasets import PDBDataset
from MLP import MLP
import matplotlib.pyplot as plt

# for PL binding


mydir_aff = '../../datasets/index/INDEX_general_PL_data.2020'
mydir = '../../datasets/refined-set'

aff_dict = get_affinities(mydir_aff)
pdb_files = read_dataset(mydir)
dataset = PDBDataset(pdb_files, aff_dict)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True)

input_dim = len(dataset[0][0])
hidden_dim = 15
output_dim = 1
model = MLP(input_dim, hidden_dim, output_dim)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 70
loss_values = []


loss_values = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = mse_loss(outputs, torch.unsqueeze(targets,-1))
        epoch_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_values.append(epoch_loss / len(dataloader))
    
print('training complete')

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    val_losses = []
    model.eval()
    for inputs, targets in test_dataset:
        y_pred = model(inputs)
        val_loss = mse_loss(y_pred[0], torch.tensor(targets))
        val_losses.append(val_loss)
print(sum(val_losses) / len(val_losses))
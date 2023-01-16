# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:46:38 2023

@author: anaso
"""
import torch
import torch.nn as nn
from findingaffinity import get_affinities
from creatingdataset import read_dataset
from pdbdataset import PDBDataset
from MLP import MLP
import matplotlib.pyplot as plt

# for PP binding

aff_dict = get_affinities('PP')
pdb_files = read_dataset('PP')[:20]

dataset = PDBDataset(pdb_files)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

model = MLP(9932*3, 3000, 1)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
num_epochs = 10
loss_values = []


for epoch in range(num_epochs):
    loss_epoch = []
    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        
        loss = mse_loss(outputs, torch.unsqueeze(targets,-1))
        loss_epoch.append(loss)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_values.append(sum(loss_epoch)/len(loss_epoch))
    
print('training complete')

fig, ax1 = plt.subplots(1, figsize=(12, 6), sharex=True)

ax1.plot([i.detach().numpy() for i in loss_values][2:])
ax1.set_ylabel("training loss")
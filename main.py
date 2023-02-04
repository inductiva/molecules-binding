# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:46:38 2023

@author: anaso
"""
import torch
from torch import nn
from molecules_binding.datasets import read_dataset
from molecules_binding.datasets import get_affinities
from molecules_binding.datasets import PDBDataset
from MLP import MLP
import matplotlib.pyplot as plt
import numpy as np

# from absl import app
from absl import flags
# from absl import logging
import sys

def plot_loss(errors_array):
    plt.plot([elem.detach().numpy() for elem in errors_array])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.savefig("../../plot")
    plt.show()

flags.DEFINE_string('aff_dir',
                    '../../datasets/index/INDEX_general_PL_data.2020',
                    'specify the path to the index of the dataset')

flags.DEFINE_string('data_dir', '../../datasets/refined-set',
                    'specify the path to the dataset')

flags.DEFINE_float('train_perc', 0.8, 'percentage of train-validation-split')

flags.DEFINE_integer('batch_s', 1, 'batch size')

flags.DEFINE_integer('hidden_size', 15, 'size of the hidden layer')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
# for PL binding

path ="../../datasetmlp"
aff_dict = get_affinities(FLAGS.aff_dir)
pdb_files = read_dataset(FLAGS.data_dir)
dataset = PDBDataset(pdb_files, aff_dict)

train_size = int(FLAGS.train_perc * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=FLAGS.batch_s,
                                         shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=FLAGS.batch_s,
                                         shuffle=False)

input_dim = len(dataset[0][0])
hidden_dim = FLAGS.hidden_size
output_dim = 1
model = MLP(input_dim, hidden_dim, output_dim)
model.double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 70


def train():
    model.train()
    for inputs, target in train_loader:
        out = model(inputs.double())
        loss = criterion(out, torch.unsqueeze(target, -1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader):
    model.eval()

    mse = 0
    for inputs, target in loader:
        out = model(inputs.double())
        mse += criterion(out, torch.unsqueeze(target, -1))
    return mse / len(loader)


train_errors = []
test_errors = []
for epoch in range(1, 50):
    model.double()
    train()
    train_err = test(train_loader)
    test_err = test(test_loader)
    print(
        f"Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
            Test Error: {test_err:.4f}"
    )
    train_errors += [train_err]
    test_errors += [test_err]

# loss_values = []
# for epoch in range(num_epochs):
#     epoch_loss = 0
#     for inputs, targets in train_loader:
#         # Forward pass
#         outputs = model(inputs)
#         loss = mse_loss(outputs, torch.unsqueeze(targets, -1))
#         epoch_loss += loss.item()

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     loss_values.append(epoch_loss / len(train_loader))

# print('training complete')

# plt.plot(loss_values)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# with torch.no_grad():
#     val_losses = []
#     model.eval()
#     for inputs, targets in test_dataset:
#         y_pred = model(inputs)
#         val_loss = mse_loss(y_pred[0], torch.as_tensor(targets))
#         val_losses.append(val_loss)
# print(sum(val_losses) / len(val_losses))

plot_loss(train_errors)
plot_loss(test_errors)

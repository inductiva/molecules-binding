# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:55:01 2023

@author: anaso
"""

from torch import nn


class MLP(nn.Module):
    """ Simple Multilayer perceptron """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

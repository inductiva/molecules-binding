# -*- coding: utf-8 -*-
'''
Created on Fri Jan 13 17:46:38 2023

@author: anaso
'''
import torch
from torch import nn
# from molecules_binding.datasets import read_dataset
# from molecules_binding.datasets import get_affinities
# from molecules_binding.datasets import PDBDataset
from MLP import MLP
import matplotlib.pyplot as plt
# import numpy as np
import pickle
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('aff_dir',
                    '../../datasets/index/INDEX_general_PL_data.2020',
                    'specify the path to the index of the dataset')

flags.DEFINE_string('data_dir', '../../datasets/refined-set',
                    'specify the path to the dataset')

flags.DEFINE_float('train_perc', 0.8, 'percentage of train-validation-split')

flags.DEFINE_integer('batch_s', 1, 'batch size')

flags.DEFINE_integer('hidden_size', 15, 'size of the hidden layer')

flags.DEFINE_string('dataset_stored', '../../datasetprocessed/mlp_dataset',
                    'dataset takes too long too parse')

flags.DEFINE_integer('num_epochs', 40, 'percentage of train-validation-split')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')


def plot_loss(errors_array):
    plt.plot([elem.detach().numpy() for elem in errors_array])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('../../plot')
    plt.show()


def main(_):

    # aff_dict = get_affinities(FLAGS.aff_dir)
    # pdb_files = read_dataset(FLAGS.data_dir)
    # dataset = PDBDataset(pdb_files, aff_dict)

    # # Save the variable to disk
    # with open(FLAGS.dataset_stored, 'wb') as file:
    #     pickle.dump(dataset, file)

    with open(FLAGS.dataset_stored, 'rb') as file:
        dataset = pickle.load(file)

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

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(dataset[0][0])
    hidden_dim = FLAGS.hidden_size
    output_dim = 1
    model = MLP(input_dim, hidden_dim, output_dim)
    # model = model.to(device)
    model.double()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    def train():
        model.train()
        for inputs, target in train_loader:
            # inputs = inputs.to(device)
            # target = target.to(device)
            out = model(inputs.double())
            loss = criterion(out, torch.unsqueeze(target, -1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        model.eval()

        mse = 0
        for inputs, target in loader:
            # inputs = inputs.to(device)
            # target = target.to(device)
            out = model(inputs.double())
            mse += criterion(out, torch.unsqueeze(target, -1))
        return mse / len(loader)

    train_errors = []
    test_errors = []
    for epoch in range(1, FLAGS.num_epochs):
        model.double()
        train()
        train_err = test(train_loader)
        test_err = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
                Test Error: {test_err:.4f}')
        train_errors += [train_err]
        test_errors += [test_err]

    plot_loss(train_errors)
    plot_loss(test_errors)


if __name__ == "__main__":
    app.run(main)

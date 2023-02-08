# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:01:43 2023

@author: anaso
"""
from molecules_binding.models import GraphNN
from molecules_binding.graphdataset import num_features
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
from absl import flags
from absl import app
import psutil
import numpy as np
from scipy.stats import spearmanr

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")

flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("train_perc", 0.8, "percentage of train-validation-split")

flags.DEFINE_integer("batch_size", 32, "batch size")

# flags.DEFINE_integer("num_hidden", 30,
#                      "size of the new features after conv layer")

flags.DEFINE_multi_integer("num_hidden", [40, 30, 30],
                           "size of the new features after conv layer")

flags.DEFINE_integer("num_epochs", 30, "number of epochs")

flags.DEFINE_float("learning_rate", 0.001, "learning rate")


def plot_loss(errors_array):
    plt.plot([elem.detach().numpy() for elem in errors_array])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, criterion, optimizer):
    model.train()
    for data in train_loader:
        data = data.to(device)
        out = model(data, data.batch)
        loss = criterion(out, data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader, model, criterion):
    model.eval()
    mse = 0
    for data in loader:
        data = data.to(device)
        out = model(data, data.batch)
        # print(out, data.y)
        mse += criterion(out, data.y.unsqueeze(1))
    return mse.detach() / len(loader)


def test_final(loader, model):
    preds = []
    reals = []
    model.eval()
    for data in loader:
        data = data.to(device)
        pred = model(data, data.batch)
        real = data.y.unsqueeze(1)
        preds += [pred.detach().numpy()[0][0]]
        reals += [real.detach().numpy()[0][0]]
    return np.array(preds), np.array(reals)


def statistics(preds, reals):
    error = np.abs(preds - reals)
    mae = np.mean(error)
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    pearson_coef = np.corrcoef(preds, reals)
    correlation, p_value = spearmanr(preds, reals)

    return mae, mse, rmse, pearson_coef, correlation, p_value


def main(_):
    dataset = torch.load(FLAGS.path_dataset)
    train_size = int(FLAGS.train_perc * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False)

    stat_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GraphNN(hidden_channels=FLAGS.num_hidden,
                    num_node_features=num_features)
    model = model.to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    criterion = torch.nn.MSELoss()

    train_errors = []
    test_errors = []
    for epoch in range(1, FLAGS.num_epochs + 1):
        train(model, train_loader, criterion, optimizer)
        print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
        train_err = test(train_loader, model, criterion)
        print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
        test_err = test(test_loader, model, criterion)
        print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
        print(f"Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
                Test Error: {test_err:.4f}")
        train_errors += [train_err]
        test_errors += [test_err]

    preds, reals = test_final(stat_loader, model)

    mae, mse, rmse, pearson_coef, spearman_corr, spearman_p_value = \
        statistics(preds, reals)

    print(f"MAE is {mae}, MSE is {mse}, RMSE is {rmse}",
          f"Pearson coefficient is {pearson_coef}",
          f"Spearman correlation is {spearman_corr}",
          f"Spearman p-value is {spearman_p_value}")

    plot_loss(train_errors)
    plot_loss(test_errors)


if __name__ == "__main__":
    app.run(main)

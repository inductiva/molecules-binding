# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:52:59 2023

@author: anaso
"""
import torch
from molecules_binding.models import GraphNN
from molecules_binding.graphdataset import num_features
from torch_geometric.loader import DataLoader
from absl import flags
from absl import app
import numpy as np
from scipy.stats import spearmanr

FLAGS = flags.FLAGS
flags.DEFINE_multi_integer("num_hidden", [50, 40, 30, 30],
                           "size of the new features after conv layer")

flags.DEFINE_string("path_dataset", "../../core_dataset",
                    "specify the path to the stored processed dataset")


def test_final(loader, model, device):
    preds = []
    reals = []
    model.eval()
    for data in loader:
        data = data.to(device)
        pred = model(data, data.batch)
        real = data.y.unsqueeze(1)
        preds += [pred.detach().cpu().numpy()[0][0]]
        reals += [real.detach().cpu().numpy()[0][0]]
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
    device = torch.device("cpu")
    model = GraphNN(hidden_channels=FLAGS.num_hidden,
                    num_node_features=num_features)
    model.load_state_dict(torch.load("../../resultados01/model"))
    model = model.to(device)
    model = model.to(float)
    # model.eval()

    stat_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    preds, reals = test_final(stat_loader, model, device)

    mae, mse, rmse, pearson_coef, spearman_corr, spearman_p_value = \
        statistics(preds, reals)

    print(f"MAE is {mae}, MSE is {mse}, RMSE is {rmse}",
          f"Pearson coefficient is {pearson_coef}",
          f"Spearman correlation is {spearman_corr}",
          f"Spearman p-value is {spearman_p_value}")


if __name__ == "__main__":
    app.run(main)

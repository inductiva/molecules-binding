# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:01:43 2023

@author: anaso
"""
from molecules_binding.models import GCN
from molecules_binding.datasets import read_dataset
from molecules_binding.graphdataset import GraphDataset
from molecules_binding.graphdataset import num_features
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
from absl import flags
from absl import app

FLAGS = flags.FLAGS
# for name in list(flags.FLAGS):
#       delattr(flags.FLAGS,name)

flags.DEFINE_string("aff_dir",
                    "../../datasets/index/INDEX_general_PL_data.2020",
                    "specify the path to the index of the dataset")

flags.DEFINE_string("data_dir", "../../datasets/refined-set",
                    "specify the path to the dataset")

# flags.DEFINE_string("path_dataset",
#                     "../../datasetprocessed/dataset_stored_new",
#                     "specify the path to the dataset")
flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")

flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("train_perc", 0.8, "percentage of train-validation-split")

flags.DEFINE_integer("batch_size", 32, "batch size")

flags.DEFINE_integer("num_hidden", 30,
                     "size of the new features after conv layer")

flags.DEFINE_integer("num_epochs", 30, "number of epochs")


# Create the dataset object and stores it in path
def create_dataset(direct: str, aff_dir: str, path: str):
    pdb_files = read_dataset(direct)
    datasetg = GraphDataset(pdb_files, aff_dir)
    torch.save(datasetg, path)


def plot_loss(errors_array):
    plt.plot([elem.detach().numpy() for elem in errors_array])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def main(_):

    # create_dataset(FLAGS.data_dir, FLAGS.aff_dir, FLAGS.path_dataset)

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

    model = GCN(hidden_channels=FLAGS.num_hidden,
                num_node_features=num_features)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()
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
            print(out[0], data.y[0])
            # print(out, data.y)
            mse += criterion(out, data.y.unsqueeze(1))
            print(criterion(out, data.y.unsqueeze(1)))
        print(mse, len(loader))
        return mse / len(loader)

    train_errors = []
    test_errors = []
    for epoch in range(1, FLAGS.num_epochs):
        train()
        train_err = test(train_loader)
        test_err = test(test_loader)
        print(f"Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
                Test Error: {test_err:.4f}")
        train_errors += [train_err]
        test_errors += [test_err]

    plot_loss(train_errors)

    plot_loss(test_errors)


if __name__ == "__main__":
    app.run(main)

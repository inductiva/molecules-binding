"""
Train the GNN
"""
from molecules_binding.models import GraphNN
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
from absl import flags
from absl import app
import numpy as np
from scipy.stats import spearmanr
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_float("train_perc", 0.8, "percentage of train-validation-split")

flags.DEFINE_integer("batch_size", 8, "batch size")

flags.DEFINE_list("num_hidden_graph", [64, 96, 128],
                  "size of message passing layers")

flags.DEFINE_list("num_hidden_linear", [], "size of linear layers")

flags.DEFINE_integer("num_epochs", 30, "number of epochs")

flags.DEFINE_float("learning_rate", 0.001, "learning rate")

flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")

flags.DEFINE_string("path_error_train", None,
                    "specify the path to store errors train")

flags.mark_flag_as_required("path_error_train")

flags.DEFINE_string("path_error_test", None,
                    "specify the path to store errors test")

flags.mark_flag_as_required("path_error_test")

flags.DEFINE_string("path_preds", None, "specify the path to predictions")

flags.mark_flag_as_required("path_preds")

flags.DEFINE_string("path_reals", None, "specify the path to real values")

flags.mark_flag_as_required("path_reals")

flags.DEFINE_string("path_plots", None, "specify the path to store plots")

flags.mark_flag_as_required("path_plots")

flags.DEFINE_string("path_model", None, "specify the path to store model")

flags.mark_flag_as_required("path_model")


def plot_loss(errors_array, path_plot):
    plt.plot([elem.detach().cpu().numpy() for elem in errors_array])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path_plot)
    plt.show()


def train(model, train_loader, criterion, optimizer, device, dropout_rate):
    model.train()
    for data in train_loader:
        data = data.to(device)
        out = model(data, data.batch, dropout_rate)
        loss = criterion(out, data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader, model, criterion, device, dropout_rate):
    model.eval()
    mse = 0
    for data in loader:
        data = data.to(device)
        out = model(data, data.batch, dropout_rate)
        # print(out, data.y)
        mse += criterion(out, data.y.unsqueeze(1)).detach()
    return mse / len(loader)


def test_final(loader, model, device, dropout_rate):
    preds = []
    reals = []
    model.eval()
    for data in loader:
        data = data.to(device)
        pred = model(data, data.batch, dropout_rate)
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


def store_list(somelist, path_list):
    with open(path_list, "wb") as fp:
        pickle.dump(somelist, fp)


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    graph_layer_sizes = list(map(int, FLAGS.num_hidden_graph))
    linear_layer_sizes = list(map(int, FLAGS.num_hidden_linear))
    model = GraphNN(dataset[0].num_node_features, graph_layer_sizes,
                    linear_layer_sizes)
    model = model.to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    criterion = torch.nn.MSELoss()

    train_errors = []
    test_errors = []
    for epoch in range(1, FLAGS.num_epochs + 1):
        train(model, train_loader, criterion, optimizer, device,
              FLAGS.dropout_rate)
        train_err = test(train_loader, model, criterion, device,
                         FLAGS.dropout_rate)
        test_err = test(test_loader, model, criterion, device,
                        FLAGS.dropout_rate)
        train_errors += [train_err]
        test_errors += [test_err]

        print(f"Epoch: {epoch:03d}, Train Error: {train_err:.4f}, \
                Test Error: {test_err:.4f}")

    store_list(train_errors, FLAGS.path_error_train)
    store_list(test_errors, FLAGS.path_error_test)

    preds, reals = test_final(stat_loader, model, device, FLAGS.dropout_rate)

    store_list(preds, FLAGS.path_preds)
    store_list(reals, FLAGS.path_reals)

    mae, mse, rmse, pearson_coef, spearman_corr, spearman_p_value = \
        statistics(preds, reals)

    print(f"MAE is {mae}, MSE is {mse}, RMSE is {rmse}",
          f"Pearson coefficient is {pearson_coef}",
          f"Spearman correlation is {spearman_corr}",
          f"Spearman p-value is {spearman_p_value}")

    torch.save(model.cpu().state_dict(), FLAGS.path_model)
    plot_loss(train_errors, FLAGS.path_plots)
    plot_loss(test_errors, FLAGS.path_plots)


if __name__ == "__main__":
    app.run(main)

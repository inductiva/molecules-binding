"""
Lightning Code
"""
import torch
from molecules_binding.models import GraphNN
from molecules_binding.callbacks import LossMonitor, MetricsMonitor
from molecules_binding.lightning_wrapper import GraphNNLightning
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from absl import flags
from absl import app
import mlflow
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import random

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_list("learning_rates", [0.1, 0.01, 0.001, 0.0001, 0.00001],
                  "the learning rates to experiment with")
flags.DEFINE_list("dropout_rates", [0, 0.1, 0.2, 0.3],
                  "the dropout rates to experiment with")
flags.DEFINE_float("train_perc", 0.9, "percentage of train-validation-split")
flags.DEFINE_integer("splitting_seed", 42, "Seed for splitting dataset")
# flags.DEFINE_list("num_hidden_graph", [64, 96, 128],
#                   "size of message passing layers")
flags.DEFINE_multi_string("dim_message_passing_layers", None,
                          "try different numbers of message passing layers")
flags.DEFINE_multi_string("dim_fully_connected_layers", None,
                          "try different numbers of linear layers")

flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_epochs", 300, "number of epochs")
flags.DEFINE_integer("num_workers", 3, "number of workers")
flags.DEFINE_boolean("use_gpu", True, "True if using gpu, False if not")
flags.DEFINE_string("comment", None, "Add a comment to the experiment.")
flags.DEFINE_integer("num_cpus_per_worker", 1,
                     "The number of cpus for each worker.")
flags.DEFINE_string("mlflow_server_uri", None,
                    "Tracking uri for mlflow experiments.")
flags.DEFINE_integer(
    "early_stopping_patience", 100,
    "How many epochs to wait for improvement before stopping.")
flags.DEFINE_boolean("shuffle", False, "Sanity Check: Shuffle labels")
flags.DEFINE_integer("shuffling_seed", 42, "Seed for shuffling labels")


def _log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def train(config, train_dataset, val_dataset, batch_size, max_epochs, comment,
          train_perc, splitting_seed, early_stopping_patience, num_workers,
          mlflow_server_uri, use_gpu):

    learning_rate = config["learning_rate"]
    message_passing_layers = config["message_passing_layers"]
    fully_connected_layers = config["fully_connected_layers"]
    dropout_rate = config["dropout_rate"]

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    model = GraphNN(train_dataset[0].num_node_features, message_passing_layers,
                    fully_connected_layers)
    model.double()
    lightning_model = GraphNNLightning(model, learning_rate, batch_size,
                                       dropout_rate)

    # Log training parameters to mlflow.
    if mlflow_server_uri is not None:
        mlflow.set_tracking_uri(mlflow_server_uri)

    mlflow.set_experiment("molecules_binding")

    with mlflow.start_run():
        _log_parameters(batch_size=batch_size,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate,
                        message_passing_layers=message_passing_layers,
                        fully_connected_layers=fully_connected_layers,
                        comment=comment,
                        data_split=train_perc,
                        num_node_features=train_dataset[0].num_node_features,
                        num_edge_features=train_dataset[0].num_edge_features,
                        early_stopping_patience=early_stopping_patience,
                        dataset_size=len(train_dataset) + len(val_dataset),
                        splitting_seed=splitting_seed)

        run_id = mlflow.active_run().info.run_id
        loss_callback = LossMonitor(run_id)
        metrics_callback = MetricsMonitor(run_id)
        # Early stopping.
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=early_stopping_patience,
            mode="min")

        report = TuneReportCallback(["loss", "val_loss"], on="validation_end")

        callbacks = [
            loss_callback, metrics_callback, early_stopping_callback, report
        ]
        print("callbacks created")
    accelerator = "gpu" if use_gpu else None

    trainer = Trainer(max_epochs=max_epochs,
                      accelerator=accelerator,
                      callbacks=callbacks,
                      logger=False)

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


def main(_):
    dataset = torch.load(FLAGS.path_dataset)
    train_size = int(FLAGS.train_perc * len(dataset))
    test_size = len(dataset) - train_size

    # Sanity Check : Shuffling labels
    if FLAGS.shuffle:
        random.seed(FLAGS.shuffling_seed)
        labels = [data.y for data in dataset]
        labels_shuffled = labels.copy()
        random.shuffle(labels_shuffled)

        for i in range(len(dataset)):
            dataset[i].y = labels_shuffled[i]

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(FLAGS.splitting_seed))

    dim_message_passing_layers = [
        list(map(int, message_passing_layers.split(" ")))
        for message_passing_layers in FLAGS.dim_message_passing_layers
    ]

    dim_fully_connected_layers = [
        list(map(int, fully_connected_layers.split(" ")))
        for fully_connected_layers in FLAGS.dim_fully_connected_layers
    ]

    config = {
        "learning_rate":
            tune.grid_search(list(map(float, FLAGS.learning_rates))),
        "message_passing_layers":
            tune.grid_search(
                list(
                    map(lambda x: list(map(int, x)),
                        dim_message_passing_layers))),
        "fully_connected_layers":
            tune.grid_search(
                list(
                    map(lambda x: list(map(int, x)),
                        dim_fully_connected_layers))),
        "dropout_rate":
            tune.grid_search(list(map(float, FLAGS.dropout_rates)))
    }

    resources_per_trial = {
        "cpu": FLAGS.num_cpus_per_worker,
        "gpu": 1 if FLAGS.use_gpu else 0
    }

    trainable = tune.with_parameters(
        train,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=FLAGS.batch_size,
        max_epochs=FLAGS.max_epochs,
        comment=FLAGS.comment,
        train_perc=FLAGS.train_perc,
        splitting_seed=FLAGS.splitting_seed,
        early_stopping_patience=FLAGS.early_stopping_patience,
        num_workers=FLAGS.num_workers,
        mlflow_server_uri=FLAGS.mlflow_server_uri,
        use_gpu=FLAGS.use_gpu)

    ray.init()
    tune.run(trainable,
             config=config,
             num_samples=1,
             resources_per_trial=resources_per_trial)


if __name__ == "__main__":
    app.run(main)

"""
Lightning Code
"""
import torch
from molecules_binding import models
from molecules_binding import callbacks as our_callbacks
from molecules_binding import lightning_wrapper
from torch_geometric import loader
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from absl import flags
from absl import app
import mlflow
import ray
from ray import tune
from ray.tune.integration import pytorch_lightning as tune_pl
import inductiva_ml

FLAGS = flags.FLAGS

flags.DEFINE_list("paths_dataset", None, "specify the path to the datasets")

flags.DEFINE_list("learning_rates", [0.1, 0.01, 0.001, 0.0001, 0.00001],
                  "the learning rates to experiment with")
flags.DEFINE_list("dropout_rates", [0, 0.1, 0.2, 0.3],
                  "the dropout rates to experiment with")
flags.DEFINE_list("weight_decays", [0, 0.0001],
                  "the weight decays to experiment with")
flags.DEFINE_multi_string("dim_embedding_layers", None,
                          "try different numbers of embedding layers")
flags.DEFINE_bool("use_message_passing", True,
                  "If set to False, this is the MLP benchmark test")
flags.DEFINE_list("n_attention_heads", [1],
                  "Number of attention heads to experiment with")
flags.DEFINE_float("train_split", 0.9, "percentage of train-validation-split")
flags.DEFINE_integer("splitting_seed", 42, "Seed for splitting dataset")
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
flags.DEFINE_boolean("use_batch_norm", False, "Use batch normalization")


def _log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def train(config, batch_size, max_epochs, comment, train_split, splitting_seed,
          early_stopping_patience, num_workers, mlflow_server_uri, use_gpu,
          use_batch_norm, use_message_passing):

    learning_rate = config["learning_rate"]
    message_passing_layers = config["message_passing_layers"]
    fully_connected_layers = config["fully_connected_layers"]
    embedding_layers = config["embedding_layers"]
    dropout_rate = config["dropout_rate"]
    weight_decay = config["weight_decay"]
    path_dataset = config["path_dataset"]
    n_attention_head = config["n_attention_head"]

    dataset = torch.load(path_dataset)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(splitting_seed))

    train_loader = loader.DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=True)
    val_loader = loader.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False)

    model = models.GraphNN(train_dataset[0].num_node_features,
                           message_passing_layers, fully_connected_layers,
                           use_batch_norm, dropout_rate, embedding_layers,
                           n_attention_head)
    model.double()
    lightning_model = lightning_wrapper.GraphNNLightning(
        model, learning_rate, batch_size, dropout_rate, weight_decay,
        use_message_passing)

    # Log training parameters to mlflow.
    if mlflow_server_uri is not None:
        mlflow.set_tracking_uri(mlflow_server_uri)

    mlflow.set_experiment("molecules_binding")

    with mlflow.start_run():
        _log_parameters(path_dataset=path_dataset,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate,
                        n_attention_head=n_attention_head,
                        message_passing_layers=message_passing_layers,
                        fully_connected_layers=fully_connected_layers,
                        embedding_layers=embedding_layers,
                        use_message_passing=use_message_passing,
                        comment=comment,
                        data_split=train_split,
                        num_node_features=train_dataset[0].num_node_features,
                        num_edge_features=train_dataset[0].num_edge_features,
                        early_stopping_patience=early_stopping_patience,
                        dataset_size=len(train_dataset) + len(val_dataset),
                        splitting_seed=splitting_seed,
                        weight_decay=weight_decay,
                        max_epochs=max_epochs,
                        use_batch_norm=use_batch_norm)

        run_id = mlflow.active_run().info.run_id
        loss_callback = our_callbacks.LossMonitor(run_id)
        metrics_callback = our_callbacks.MetricsMonitor(run_id)
        gpu_usage_callback = inductiva_ml.callbacks.GPUUsage(run_id)
        # Early stopping.
        early_stopping_callback = pl_callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=early_stopping_patience,
            mode="min")

        report = tune_pl.TuneReportCallback(["loss", "val_loss"],
                                            on="validation_end")

        callbacks = [
            loss_callback, metrics_callback, early_stopping_callback, report,
            gpu_usage_callback
        ]

    accelerator = "gpu" if use_gpu else None

    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator=accelerator,
                         callbacks=callbacks,
                         logger=False)

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


def main(_):

    dim_message_passing_layers = [
        list(map(int, message_passing_layers.split(" ")))
        for message_passing_layers in FLAGS.dim_message_passing_layers
    ]

    dim_fully_connected_layers = [
        list(map(int, fully_connected_layers.split(" ")))
        for fully_connected_layers in FLAGS.dim_fully_connected_layers
    ]

    dim_embedding_layers = [
        list(map(int, embedding_layers.split(" ")))
        for embedding_layers in FLAGS.dim_embedding_layers
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
        "embedding_layers":
            tune.grid_search(
                list(map(lambda x: list(map(int, x)), dim_embedding_layers))),
        "dropout_rate":
            tune.grid_search(list(map(float, FLAGS.dropout_rates))),
        "weight_decay":
            tune.grid_search(list(map(float, FLAGS.weight_decays))),
        "n_attention_head":
            tune.grid_search(list(map(int, FLAGS.n_attention_heads))),
        "path_dataset":
            tune.grid_search(list(map(str, FLAGS.paths_dataset)))
    }

    resources_per_trial = {
        "cpu": FLAGS.num_cpus_per_worker,
        "gpu": 1 if FLAGS.use_gpu else 0
    }

    trainable = tune.with_parameters(
        train,
        batch_size=FLAGS.batch_size,
        max_epochs=FLAGS.max_epochs,
        comment=FLAGS.comment,
        train_split=FLAGS.train_split,
        splitting_seed=FLAGS.splitting_seed,
        early_stopping_patience=FLAGS.early_stopping_patience,
        num_workers=FLAGS.num_workers,
        mlflow_server_uri=FLAGS.mlflow_server_uri,
        use_gpu=FLAGS.use_gpu,
        use_batch_norm=FLAGS.use_batch_norm,
        use_message_passing=FLAGS.use_message_passing)

    ray.init()
    tune.run(trainable,
             config=config,
             num_samples=1,
             resources_per_trial=resources_per_trial)


if __name__ == "__main__":
    app.run(main)

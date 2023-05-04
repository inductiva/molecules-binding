"""
Lightning Code
"""
import torch
from molecules_binding.models import GraphNN
from molecules_binding.callbacks import LossMonitor, MetricsMonitor
from molecules_binding.lightning_wrapper import GraphNNLightning
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from absl import flags
from absl import app
import mlflow
from ray_lightning import RayStrategy
import ray

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
flags.DEFINE_float("train_perc", 0.9, "percentage of train-validation-split")
flags.DEFINE_list("num_hidden_graph", [64, 96, 128],
                  "size of message passing layers")
flags.DEFINE_list("num_hidden_linear", [], "size of linear layers")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_epochs", 300, "number of epochs")
flags.DEFINE_integer("num_workers", 3, "number of workers")
flags.DEFINE_boolean("use_gpu", True, "True if using gpu, False if not")
flags.DEFINE_string("add_comment", None, "Add a comment to the experiment.")
# Flags for Ray Training
flags.DEFINE_boolean("use_ray", False, "Controls if it uses ray")
flags.DEFINE_integer("num_cpus_per_worker", 1,
                     "The number of cpus for each worker.")
flags.DEFINE_string("mlflow_server_uri", None,
                    "Tracking uri for mlflow experiments.")


def _log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def main(_):
    dataset = torch.load(FLAGS.path_dataset)
    train_size = int(FLAGS.train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=FLAGS.batch_size,
                            num_workers=FLAGS.num_workers,
                            shuffle=False)

    graph_layer_sizes = list(map(int, FLAGS.num_hidden_graph))
    linear_layer_sizes = list(map(int, FLAGS.num_hidden_linear))
    model = GraphNN(dataset[0].num_node_features, graph_layer_sizes,
                    linear_layer_sizes)
    model.double()

    lightning_model = GraphNNLightning(model, FLAGS.learning_rate,
                                       FLAGS.batch_size, FLAGS.dropout_rate)

    # Log training parameters to mlflow.
    if FLAGS.mlflow_server_uri is not None:
        mlflow.set_tracking_uri(FLAGS.mlflow_server_uri)

    mlflow.set_experiment("lightning_graphnet")

    with mlflow.start_run():
        _log_parameters(batch_size=FLAGS.batch_size,
                        learning_rate=FLAGS.learning_rate,
                        dropout_rate=FLAGS.dropout_rate,
                        num_hidden_graph=FLAGS.num_hidden_graph,
                        num_hidden_linear=FLAGS.num_hidden_linear,
                        comment=FLAGS.add_comment,
                        data_split=FLAGS.train_perc,
                        num_node_features=dataset[0].num_node_features,
                        num_edge_features=dataset[0].num_edge_features)
        run_id = mlflow.active_run().info.run_id
        loss_callback = LossMonitor(run_id)
        metrics_callback = MetricsMonitor(run_id)
        callbacks = [loss_callback, metrics_callback]

    if FLAGS.use_ray:
        ray.init()

        plugin = RayStrategy(num_workers=FLAGS.num_workers,
                             num_cpus_per_worker=FLAGS.num_cpus_per_worker,
                             use_gpu=FLAGS.use_gpu)
        trainer = Trainer(max_epochs=FLAGS.max_epochs,
                          strategy=plugin,
                          logger=False,
                          callbacks=callbacks)
    else:
        accelerator = "gpu" if FLAGS.use_gpu else None
        trainer = Trainer(max_epochs=FLAGS.max_epochs,
                          accelerator=accelerator,
                          callbacks=callbacks)

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(main)

"""
Lightning Code
"""
import torch
from molecules_binding.models import MLP
from molecules_binding.callbacks import LossMonitor, MetricsMonitor
from molecules_binding.lightning_wrapper import MLPLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from absl import flags
from absl import app
import mlflow

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
flags.DEFINE_multi_integer("num_hidden", [128, 128],
                           "size of the new features after conv layer")
flags.DEFINE_float("train_split", 0.8, "percentage of train-validation-split")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_epochs", 100, "number of epochs")
flags.DEFINE_integer("num_workers", 12, "number of workers")
flags.DEFINE_bool("use_gpu", True, "True if using gpu, False if not")
flags.DEFINE_string("add_comment", None, "Add a comment to the experiment.")
flags.DEFINE_string("mlflow_server_uri", None,
                    "Tracking uri for mlflow experiments.")
flags.DEFINE_integer(
    "early_stopping_patience", 100,
    "How many epochs to wait for improvement before stopping.")

flags.DEFINE_bool("shuffle_nodes", False, "Sanity Check: Shuffle nodes")


def _log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    train_size = int(FLAGS.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42))

    if FLAGS.shuffle_nodes:
        for i in val_dataset.indices:
            dataset.shuffle_nodes(i)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=FLAGS.batch_size,
                                               num_workers=FLAGS.num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=FLAGS.batch_size,
                                             num_workers=FLAGS.num_workers,
                                             shuffle=False)

    layer_sizes = list(map(int, FLAGS.num_hidden))
    model = MLP(len(dataset[0][0]), layer_sizes, 1)
    model.double()

    lightning_model = MLPLightning(model, FLAGS.learning_rate)

    # Log training parameters to mlflow.
    if FLAGS.mlflow_server_uri is not None:
        mlflow.set_tracking_uri(FLAGS.mlflow_server_uri)

    mlflow.set_experiment("lightning_mlp")

    with mlflow.start_run():
        _log_parameters(batch_size=FLAGS.batch_size,
                        learning_rate=FLAGS.learning_rate,
                        num_hidden=FLAGS.num_hidden,
                        comment=FLAGS.add_comment,
                        first_layer_size=len(dataset[0][0]),
                        early_stopping_patience=FLAGS.early_stopping_patience,
                        data_split=FLAGS.train_split,
                        dataset=str(FLAGS.path_dataset))
        run_id = mlflow.active_run().info.run_id
        loss_callback = LossMonitor(run_id)
        metrics_callback = MetricsMonitor(run_id)
        # Early stopping.
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=FLAGS.early_stopping_patience,
            mode="min")
        callbacks = [loss_callback, metrics_callback, early_stopping_callback]

    trainer = Trainer(max_epochs=FLAGS.max_epochs,
                      accelerator="gpu" if FLAGS.use_gpu else None,
                      devices=1,
                      callbacks=callbacks,
                      log_every_n_steps=15)
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(main)

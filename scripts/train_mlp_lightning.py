"""
Lightning Code
"""
import torch
from molecules_binding import models
from molecules_binding import callbacks as our_callbacks
from molecules_binding import lightning_wrapper
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from absl import flags
from absl import app
import mlflow

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")
flags.DEFINE_float("weight_decay", 0, "Weight decay")
flags.DEFINE_bool("use_batch_norm", False,
                  "True if using batch norm, False if not")
flags.DEFINE_multi_integer("num_hidden", [128, 128],
                           "size of the new features after conv layer")
flags.DEFINE_float("train_split", 0.8, "percentage of train-validation-split")
flags.DEFINE_integer("splitting_seed", 42, "Seed for splitting dataset")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_epochs", 100, "number of epochs")
flags.DEFINE_integer("num_workers", 12, "number of workers")
flags.DEFINE_bool("use_gpu", True, "True if using gpu, False if not")
flags.DEFINE_string("comment", None, "Add a comment to the experiment.")
flags.DEFINE_string("mlflow_server_uri", None,
                    "Tracking uri for mlflow experiments.")
flags.DEFINE_integer(
    "early_stopping_patience", 100,
    "How many epochs to wait for improvement before stopping.")

flags.DEFINE_bool("shuffle_nodes", False, "Sanity Check: Shuffle nodes")
flags.DEFINE_bool("shuffle_all_nodes", False, "Sanity Check: Shuffle all nodes")
flags.DEFINE_bool("translate_complex", False, "Sanity Check: Translate complex")


def _log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def main(_):
    dataset = torch.load(FLAGS.path_dataset)
    train_size = int(FLAGS.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(FLAGS.splitting_seed))

    if FLAGS.shuffle_nodes:
        for i in val_dataset.indices:
            dataset.shuffle_nodes(i)

    if FLAGS.shuffle_all_nodes:
        for i in range(len(dataset)):
            dataset.shuffle_nodes(i)

    if FLAGS.translate_complex:
        for i in val_dataset.indices:
            dataset.translate_complex(i)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=FLAGS.batch_size,
                                               num_workers=FLAGS.num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=FLAGS.batch_size,
                                             num_workers=FLAGS.num_workers,
                                             shuffle=False)

    layer_sizes = list(map(int, FLAGS.num_hidden))
    model = models.MLP(len(dataset[0][0]), layer_sizes, 1, FLAGS.use_batch_norm,
                       FLAGS.dropout_rate)
    model.double()

    lightning_model = lightning_wrapper.MLPLightning(model, FLAGS.learning_rate,
                                                     FLAGS.weight_decay)

    # Log training parameters to mlflow.
    if FLAGS.mlflow_server_uri is not None:
        mlflow.set_tracking_uri(FLAGS.mlflow_server_uri)

    mlflow.set_experiment("molecules_binding")

    with mlflow.start_run():
        _log_parameters(model="MLP",
                        batch_size=FLAGS.batch_size,
                        learning_rate=FLAGS.learning_rate,
                        weight_decay=FLAGS.weight_decay,
                        use_batch_norm=FLAGS.use_batch_norm,
                        num_hidden=FLAGS.num_hidden,
                        comment=FLAGS.comment,
                        first_layer_size=len(dataset[0][0]),
                        early_stopping_patience=FLAGS.early_stopping_patience,
                        data_split=FLAGS.train_split,
                        dataset=str(FLAGS.path_dataset),
                        splitting_seed=FLAGS.splitting_seed,
                        sanity_check_shuffle_nodes=FLAGS.shuffle_nodes,
                        sanity_check_translate_complex=FLAGS.translate_complex,
                        sanity_check_shuffle_all_nodes=FLAGS.shuffle_all_nodes)

        run_id = mlflow.active_run().info.run_id
        loss_callback = our_callbacks.LossMonitor(run_id)
        metrics_callback = our_callbacks.MetricsMonitor(run_id)
        # Early stopping.
        early_stopping_callback = pl_callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=FLAGS.early_stopping_patience,
            mode="min")
        callbacks = [loss_callback, metrics_callback, early_stopping_callback]

    trainer = pl.Trainer(max_epochs=FLAGS.max_epochs,
                         accelerator="gpu" if FLAGS.use_gpu else None,
                         devices=1,
                         callbacks=callbacks,
                         log_every_n_steps=15)
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(main)

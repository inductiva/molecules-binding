"""
Lightning Code
"""
import torch
from molecules_binding import models
from molecules_binding import callbacks as our_callbacks
from molecules_binding import lightning_wrapper
from molecules_binding import parsers
from torch_geometric import loader
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from absl import flags
from absl import app
import mlflow
import ray_lightning
import ray
import random
import inductiva_ml

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
flags.DEFINE_list(
    "embedding_layers", None, "set to None if not using embedding,"
    "else specify the size of embedding layers")
flags.DEFINE_bool("use_message_passing", True,
                  "If set to False, this is the MLP benchmark test")
flags.DEFINE_integer("n_attention_heads", 1, "Number of attention heads")
flags.DEFINE_float("train_split", 0.9, "percentage of train-validation-split")
flags.DEFINE_integer("splitting_seed", 42, "Seed for splitting dataset")
flags.DEFINE_list("num_hidden_graph", [64, 96, 128],
                  "size of message passing layers")
flags.DEFINE_bool("normalize_edges", False, "Normalize edges")
flags.DEFINE_list("num_hidden_linear", [], "size of linear layers")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_epochs", 300, "number of epochs")
flags.DEFINE_integer("num_workers", 3, "number of workers")
flags.DEFINE_boolean("use_gpu", True, "True if using gpu, False if not")
flags.DEFINE_string("comment", None, "Add a comment to the experiment.")
# Flags for Ray Training
flags.DEFINE_boolean("use_ray", False, "Controls if it uses ray")
flags.DEFINE_integer("num_cpus_per_worker", 1,
                     "The number of cpus for each worker.")
flags.DEFINE_string("mlflow_server_uri", None,
                    "Tracking uri for mlflow experiments.")
flags.DEFINE_integer(
    "early_stopping_patience", 100,
    "How many epochs to wait for improvement before stopping.")
flags.DEFINE_boolean("shuffle", False, "Sanity Check: Shuffle labels")
flags.DEFINE_integer("shuffling_seed", 42, "Seed for shuffling labels")
flags.DEFINE_boolean("sanity_check_rotation", False,
                     "Sanity Check: Rotate the graph")
flags.DEFINE_list("rotation_angles", [30, 30, 30],
                  "Rotation angles if doing rotation sanity check")
flags.DEFINE_boolean("comparing_with_mlp", False,
                     "Sanity Check: Compare with MLP")
flags.DEFINE_bool("shuffle_nodes", False, "Sanity Check: Shuffle nodes")
flags.DEFINE_bool("remove_coords", False,
                  "remove coordinates of nodes, only for old dataset")
flags.DEFINE_float("weight_decay", 0, "value of weight decay")
flags.DEFINE_bool("use_batch_norm", True, "use batch norm")
flags.DEFINE_enum("which_gnn_model", "GATGNN",
                  ["GATGNN", "NodeEdgeGNN", "SeparateEdgesGNN"],
                  "which model to use")
flags.DEFINE_integer("num_processing_steps", 1, "number of processor layers")
flags.DEFINE_integer("size_processing_steps", 128, "size of processor layers")
flags.DEFINE_bool("save_model", False, "save best model")


def _log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    # remove elements from the core set
    dataset.remove_graph_by_ids(parsers.CASF_2016_core_set)

    if FLAGS.which_gnn_model == "NodeEdgeGNN":
        for graph in dataset:
            if torch.isnan(graph.edge_attr).any():
                dataset.remove_graph(dataset.data_list.index(graph))

    elif FLAGS.which_gnn_model == "SeparateEdgesGNN":
        for graph in dataset:
            if torch.isnan(graph.edge_attr_2).any():
                dataset.remove_graph(dataset.data_list.index(graph))

    train_size = int(FLAGS.train_split * len(dataset))
    test_size = len(dataset) - train_size

    if FLAGS.normalize_edges:
        for data in dataset:
            data.edge_attr[:, -8] = data.edge_attr[:, -8] * 0.1
            data.edge_attr[:, -6:-3] = data.edge_attr[:, -6:-3] * 0.1
            data.edge_attr[:, -5] = data.edge_attr[:, -5] * 0.1
            data.edge_attr[:, -2] = data.edge_attr[:, -2] * 0.1

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

    if FLAGS.shuffle_nodes:
        for i in val_dataset.indices:
            dataset.shuffle_nodes(i)

    if FLAGS.comparing_with_mlp:
        for i in range(len(dataset)):
            dataset[i].edge_attr = None

    if FLAGS.remove_coords:
        for i in range(len(dataset)):
            dataset.remove_coords_from_nodes(i)

    # only for previous representation of graphs
    if FLAGS.sanity_check_rotation:
        rotation_angles = list(map(int, FLAGS.rotation_angles))
        for i in val_dataset.indices:
            dataset.rotate_graph(i, rotation_angles, FLAGS.remove_coords)

    train_loader = loader.DataLoader(train_dataset,
                                     batch_size=FLAGS.batch_size,
                                     num_workers=FLAGS.num_workers,
                                     shuffle=True)
    val_loader = loader.DataLoader(val_dataset,
                                   batch_size=FLAGS.batch_size,
                                   num_workers=FLAGS.num_workers,
                                   shuffle=False)

    graph_layer_sizes = list(map(int, FLAGS.num_hidden_graph))
    linear_layer_sizes = list(map(int, FLAGS.num_hidden_linear))

    if FLAGS.embedding_layers is None:
        embedding_layer_sizes = None
    else:
        embedding_layer_sizes = list(map(int, FLAGS.embedding_layers))

    if FLAGS.which_gnn_model == "GATGNN":
        model = models.GraphNN(dataset[0].num_node_features, graph_layer_sizes,
                               linear_layer_sizes, FLAGS.use_batch_norm,
                               FLAGS.dropout_rate, embedding_layer_sizes,
                               FLAGS.n_attention_heads)
    elif FLAGS.which_gnn_model == "NodeEdgeGNN":
        num_edge_features = dataset[0].num_edge_features
        model = models.NodeEdgeGNN(dataset[0].num_node_features,
                                   num_edge_features, linear_layer_sizes,
                                   FLAGS.use_batch_norm, FLAGS.dropout_rate,
                                   embedding_layer_sizes,
                                   FLAGS.size_processing_steps,
                                   FLAGS.num_processing_steps)
    elif FLAGS.which_gnn_model == "SeparateEdgesGNN":
        num_edge_features = dataset[0].edge_attr_2.shape[1]
        model = models.SeparateEdgesGNN(
            dataset[0].num_node_features, num_edge_features, linear_layer_sizes,
            FLAGS.use_batch_norm, FLAGS.dropout_rate, embedding_layer_sizes,
            FLAGS.size_processing_steps, FLAGS.num_processing_steps,
            FLAGS.n_attention_heads, graph_layer_sizes)

    lightning_model = lightning_wrapper.GraphNNLightning(
        model, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.dropout_rate,
        FLAGS.weight_decay, FLAGS.use_message_passing)

    # Log training parameters to mlflow.
    if FLAGS.mlflow_server_uri is not None:
        mlflow.set_tracking_uri(FLAGS.mlflow_server_uri)

    mlflow.set_experiment("molecules_binding")

    with mlflow.start_run():
        _log_parameters(model="GNN",
                        batch_size=FLAGS.batch_size,
                        learning_rate=FLAGS.learning_rate,
                        dropout_rate=FLAGS.dropout_rate,
                        weight_decay=FLAGS.weight_decay,
                        embedding_layers=FLAGS.embedding_layers,
                        use_message_passing=FLAGS.use_message_passing,
                        n_attention_heads=FLAGS.n_attention_heads,
                        num_hidden_graph=FLAGS.num_hidden_graph,
                        num_hidden_linear=FLAGS.num_hidden_linear,
                        comment=FLAGS.comment,
                        data_split=FLAGS.train_split,
                        num_node_features=dataset[0].num_node_features,
                        num_edge_features=num_edge_features,
                        early_stopping_patience=FLAGS.early_stopping_patience,
                        dataset_size=len(dataset),
                        splitting_seed=FLAGS.splitting_seed,
                        dataset=str(FLAGS.path_dataset),
                        shuffle_nodes=FLAGS.shuffle_nodes,
                        remove_coords=FLAGS.remove_coords,
                        comparing_with_mlp=FLAGS.comparing_with_mlp,
                        use_batch_norm=FLAGS.use_batch_norm,
                        which_gnn_model=FLAGS.which_gnn_model,
                        num_processing_steps=FLAGS.num_processing_steps,
                        size_processing_steps=FLAGS.size_processing_steps,
                        max_epochs=FLAGS.max_epochs,
                        save_model=FLAGS.save_model)

        run_id = mlflow.active_run().info.run_id
        loss_callback = our_callbacks.LossMonitor(run_id)
        metrics_callback = our_callbacks.MetricsMonitor(run_id)
        gpu_usage_callback = inductiva_ml.callbacks.GPUUsage(run_id)
        # Early stopping.
        early_stopping_callback = pl_callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=FLAGS.early_stopping_patience,
            mode="min")

        callbacks = [
            loss_callback, metrics_callback, early_stopping_callback,
            gpu_usage_callback
        ]

        if FLAGS.save_model:
            checkpoint_callback = our_callbacks.MlflowBestModelsCheckpoint(
                run_id=run_id,
                monitor_metrics=[("val_loss", "min")],
                save_dir="checkpoints")
            callbacks.append(checkpoint_callback)

    if FLAGS.use_ray:
        ray.init()

        plugin = ray_lightning.RayStrategy(
            num_workers=FLAGS.num_workers,
            num_cpus_per_worker=FLAGS.num_cpus_per_worker,
            use_gpu=FLAGS.use_gpu)
        trainer = pl.Trainer(max_epochs=FLAGS.max_epochs,
                             strategy=plugin,
                             logger=False,
                             callbacks=callbacks,
                             log_every_n_steps=20)
    else:
        accelerator = "gpu" if FLAGS.use_gpu else None
        trainer = pl.Trainer(max_epochs=FLAGS.max_epochs,
                             accelerator=accelerator,
                             callbacks=callbacks)

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(main)

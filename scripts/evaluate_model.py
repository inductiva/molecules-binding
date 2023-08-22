"""given a model, evaluate it on the test set"""
import torch
from molecules_binding import models
from molecules_binding import lightning_wrapper
from torch_geometric import loader
from absl import flags
from absl import app
import mlflow

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
flags.DEFINE_enum("which_gnn_model", "GATGNN", ["GATGNN", "NodeEdgeGNN"],
                  "which model to use")
flags.DEFINE_integer("num_processing_steps", 1, "number of processor layers")
flags.DEFINE_integer("size_processing_steps", 128, "size of processor layers")
flags.DEFINE_string("path_checkpoint", None, "path to checkpoint")


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    dataset_loader = loader.DataLoader(dataset,
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
        model = models.NodeEdgeGNN(dataset[0].num_node_features,
                                   dataset[0].num_edge_features,
                                   linear_layer_sizes, FLAGS.use_batch_norm,
                                   FLAGS.dropout_rate, embedding_layer_sizes,
                                   FLAGS.size_processing_steps,
                                   FLAGS.num_processing_steps)

    lightning_model = lightning_wrapper.GraphNNLightning.load_from_checkpoint(
        FLAGS.path_checkpoint,
        model=model,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        dropout_rate=FLAGS.dropout_rate,
        weight_decay=FLAGS.weight_decay,
        use_message_passing=FLAGS.use_message_passing)
    print("lightning model", lightning_model)

    if FLAGS.mlflow_server_uri is not None:
        mlflow.set_tracking_uri(FLAGS.mlflow_server_uri)

    mlflow.set_experiment("molecules_binding")

    lightning_model.eval()
    with torch.no_grad():
        for data in dataset_loader:
            # preds = lightning_model.model(data, data.batch,
            # FLAGS.dropout_rate, FLAGS.use_message_passing)
            # print("preds", preds.squeeze())
            # print("labels", data.y[0])
            # labels = data.y[0].squeeze()
            # loss = lightning_model.criterion(labels, preds.squeeze())
            # print("loss", loss)
            # mae = torch.nn.functional.l1_loss(preds.squeeze(), labels)
            # print("mae", mae)
            # rmse = torch.sqrt(loss)
            # print("rmse", rmse)
            statistics = lightning_model.compute_statistics(data,
                                                            training=False)
            print("statistics", statistics)




if __name__ == "__main__":
    app.run(main)

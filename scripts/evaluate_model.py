"""given a model, evaluate it on the test set"""
import torch
from molecules_binding import models
from molecules_binding import lightning_wrapper
from torch_geometric import loader
from absl import flags
from absl import app
import mlflow
import tempfile

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")

flags.DEFINE_integer("batch_size", 285, "batch size")

flags.DEFINE_string("mlflow_server_uri", None,
                    "Tracking uri for mlflow experiments.")

flags.DEFINE_integer("num_workers", 12, "number of workers for dataloader")
flags.DEFINE_string("run_id", None, "run id of the model to evaluate")


def string_to_int_list(some_string):
    return [int(x.strip(" '")) for x in some_string.strip("[]").split(",")]


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    mlflow.set_tracking_uri(FLAGS.mlflow_server_uri)
    run = mlflow.get_run(FLAGS.run_id)
    parameters = run.data.params

    with tempfile.TemporaryDirectory() as temp_dir:
        path = mlflow.artifacts.download_artifacts(
            run_id=FLAGS.run_id,
            artifact_path="checkpoints/best_val_loss_model_" +
            f"{FLAGS.run_id}.ckpt",
            dst_path=temp_dir)

        dataset_loader = loader.DataLoader(dataset,
                                           batch_size=int(
                                               parameters["batch_size"]),
                                           num_workers=FLAGS.num_workers,
                                           shuffle=False)

        if parameters["which_gnn_model"] == "GATGNN":
            model = models.GraphNN(
                num_node_features=dataset[0].num_node_features,
                layer_sizes_graph=string_to_int_list(
                    parameters["num_hidden_graph"]),
                layer_sizes_linear=string_to_int_list(
                    parameters["num_hidden_linear"]),
                use_batch_norm=int(parameters["use_batch_norm"]),
                dropout_rate=int(parameters["dropout_rate"]),
                embedding_layers=string_to_int_list(
                    parameters["embedding_layers"]),
                n_attention_heads=int(parameters["n_attention_heads"]))

        elif parameters["which_gnn_model"] == "NodeEdgeGNN":
            model = models.NodeEdgeGNN(
                num_node_features=int(parameters["num_node_features"]),
                num_edge_features=int(parameters["num_edge_features"]),
                layer_sizes_linear=string_to_int_list(
                    parameters["num_hidden_linear"]),
                use_batch_norm=bool(parameters["use_batch_norm"]),
                dropout_rate=float(parameters["dropout_rate"]),
                embedding_layers=string_to_int_list(
                    parameters["embedding_layers"]),
                latent_size=int(parameters["size_processing_steps"]),
                num_processing_steps=int(parameters["num_processing_steps"]))


        lightning_model =\
            lightning_wrapper.GraphNNLightning.load_from_checkpoint(
            path,
            model=model,
            learning_rate=float(parameters["learning_rate"]),
            batch_size=int(parameters["batch_size"]),
            dropout_rate=float(parameters["dropout_rate"]),
            weight_decay=float(parameters["weight_decay"]),
            use_message_passing=bool(parameters["use_message_passing"]))
        print("lightning model", lightning_model)

        lightning_model.eval()

    with torch.no_grad():
        for data in dataset_loader:
            data1 = data.clone()
            predictions = lightning_model.model(
                data1, data1.batch, float(parameters["dropout_rate"]),
                bool(parameters["use_message_passing"]))

            labels = data.y[0].unsqueeze(-1)
            concatenation = torch.cat((predictions, labels), dim=1)

            torch.save(concatenation, "../results/" + FLAGS.run_id)


if __name__ == "__main__":
    app.run(main)

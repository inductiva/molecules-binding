"""
Lightning Code
"""
import torch
from molecules_binding.models import GraphNN
from molecules_binding.parsers import num_features
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("path_dataset", None,
                    "specify the path to the stored processed dataset")
flags.mark_flag_as_required("path_dataset")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
flags.DEFINE_multi_integer("num_hidden", [40, 30, 30, 40],
                           "size of the new features after conv layer")
flags.DEFINE_float("train_perc", 0.8, "percentage of train-validation-split")
flags.DEFINE_integer("batch_size", 32, "batch size")

criterion = torch.nn.MSELoss()


class AffinityBindingTrainer(pl.LightningModule):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, model, learning_rate, train_dataset, batch_size):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset
        self.batch_size = batch_size

    def training_step(self, data, _):
        labels = data.y.unsqueeze(1)
        outputs = self.model(data, data.batch, FLAGS.dropout_rate)
        loss = criterion(labels, outputs)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=FLAGS.learning_rate)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        return train_loader


def main(_):
    dataset = torch.load(FLAGS.path_dataset)
    train_size = int(FLAGS.train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42))

    trainer = Trainer(fast_dev_run=False)
    model = GraphNN(FLAGS.num_hidden, num_features)
    model.double()
    lightning_model = AffinityBindingTrainer(model, FLAGS.learning_rate,
                                             train_dataset, FLAGS.batch_size)
    trainer.fit(lightning_model)


if __name__ == "__main__":
    app.run(main)

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
flags.DEFINE_integer("num_epochs", 100, "number of epochs")

criterion = torch.nn.MSELoss()


class AffinityBindingTrainer(pl.LightningModule):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, model, learning_rate, dataset, batch_size, train_perc):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        train_size = int(train_perc * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42))

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
                                  num_workers=12,
                                  shuffle=True)
        return train_loader

    def validation_step(self, data, _):
        labels = data.y.unsqueeze(1)
        outputs = self.model(data, data.batch, FLAGS.dropout_rate)
        loss = criterion(labels, outputs)
        return {"val_loss": loss}

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=12,
                                shuffle=False)
        return val_loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    trainer = Trainer(fast_dev_run=False,
                      max_epochs=FLAGS.num_epochs,
                      accelerator="gpu",
                      devices=1)
    model = GraphNN(FLAGS.num_hidden, num_features)
    model.double()
    lightning_model = AffinityBindingTrainer(model, FLAGS.learning_rate,
                                             dataset, FLAGS.batch_size,
                                             FLAGS.train_perc)
    trainer.fit(lightning_model)


if __name__ == "__main__":
    app.run(main)

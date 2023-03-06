"""
Lightning Code
"""
import torch
from molecules_binding.models import MLP
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
flags.DEFINE_multi_integer("num_hidden", [128, 128],
                           "size of the new features after conv layer")
flags.DEFINE_float("train_perc", 0.8, "percentage of train-validation-split")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("num_epochs", 100, "number of epochs")

criterion = torch.nn.MSELoss()


class AffinityBinding(pl.LightningModule):
    """
        Multilayer Perceptron
    """

    def __init__(self, model, learning_rate, batch_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def training_step(self, data, _):
        inputs, labels = data
        inputs = inputs.double()
        labels = torch.unsqueeze(labels, -1).double()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=FLAGS.learning_rate)

    def validation_step(self, data, _):
        inputs, labels = data
        inputs = inputs.double()
        labels = torch.unsqueeze(labels, -1).double()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    train_size = int(FLAGS.train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=FLAGS.batch_size,
                                               num_workers=12,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=FLAGS.batch_size,
                                             num_workers=12,
                                             shuffle=False)

    layer_sizes = list(map(int, FLAGS.num_hidden))
    model = MLP(len(dataset[0][0]), layer_sizes, 1)
    model.double()

    lightning_model = AffinityBinding(model, FLAGS.learning_rate,
                                      FLAGS.batch_size)
    trainer = Trainer(fast_dev_run=False,
                      max_epochs=FLAGS.num_epochs,
                      accelerator="gpu",
                      devices=1)
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(main)

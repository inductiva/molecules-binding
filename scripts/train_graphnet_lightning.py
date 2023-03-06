"""
Lightning Code
"""
import torch
from molecules_binding.models import GraphNN
from molecules_binding.parsers import num_features
from molecules_binding.lightning_wrapper import AffinityBinding
from torch_geometric.loader import DataLoader
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


def main(_):
    dataset = torch.load(FLAGS.path_dataset)

    train_size = int(FLAGS.train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              num_workers=12,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=FLAGS.batch_size,
                            num_workers=12,
                            shuffle=False)

    model = GraphNN(FLAGS.num_hidden, num_features)
    model.double()

    lightning_model = AffinityBinding(model, FLAGS.learning_rate,
                                      FLAGS.batch_size, FLAGS.dropout_rate)
    trainer = Trainer(fast_dev_run=False,
                      max_epochs=FLAGS.num_epochs,
                      accelerator="gpu",
                      devices=1)
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(main)
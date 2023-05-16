"""
Lightning Code
"""
import torch
import pytorch_lightning as pl
from torchmetrics import SpearmanCorrCoef
from torchmetrics import PearsonCorrCoef
from torch.nn.functional import l1_loss

spearman = SpearmanCorrCoef(num_outputs=1)
pearson = PearsonCorrCoef(num_outputs=1)


class GraphNNLightning(pl.LightningModule):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, model, learning_rate, batch_size, dropout_rate):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.criterion = torch.nn.MSELoss()

    def compute_statistics(self, data, training):
        labels = data.y
        outputs = self.model(data, data.batch, self.dropout_rate).squeeze()

        if training:
            loss = self.criterion(labels, outputs)
            return loss, None, None, None, None

        loss = self.criterion(labels, outputs)
        mae = l1_loss(outputs, labels)
        rmse = torch.sqrt(loss)
        pearson_correlation = pearson(outputs.cpu(), labels.cpu())
        spearman_correlation = spearman(outputs, labels)
        return loss, mae, rmse, pearson_correlation, spearman_correlation

    def training_step(self, data, _):
        loss, _, _, _, _ = self.compute_statistics(data, training=True)
        self.log("loss", loss, batch_size=self.batch_size, on_epoch=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, data, _):
        (val_loss, mae, rmse, pearson_correlation,
         spearman_correlation) = self.compute_statistics(data, training=False)
        self.log("val_loss",
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size)
        self.log("val_mae",
                 mae,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 batch_size=self.batch_size)
        self.log("val_rmse",
                 rmse,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 batch_size=self.batch_size)
        self.log("val_pearson",
                 pearson_correlation,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 batch_size=self.batch_size)
        self.log("val_spearman",
                 spearman_correlation,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 batch_size=self.batch_size)
        return {"val_loss": val_loss}


class MLPLightning(pl.LightningModule):
    """
        Multilayer Perceptron
    """

    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()

    def compute_statistics(self, data, training):
        inputs, labels = data
        inputs = inputs.double()
        labels = labels.double()
        outputs = self.model(inputs).squeeze()

        if training:
            loss = self.criterion(labels, outputs)
            return loss, None, None, None, None

        loss = self.criterion(labels, outputs)
        mae = l1_loss(outputs, labels)
        rmse = torch.sqrt(loss)
        pearson_correlation = pearson(outputs.cpu(), labels.cpu())
        spearman_correlation = spearman(outputs, labels)
        return loss, mae, rmse, pearson_correlation, spearman_correlation

    def training_step(self, data, _):
        loss, _, _, _, _ = self.compute_statistics(data, training=True)
        self.log("loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, data, _):
        (val_loss, mae, rmse, pearson_correlation,
         spearman_correlation) = self.compute_statistics(data, training=False)
        self.log("val_loss",
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_pearson",
                 pearson_correlation,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False)
        self.log("val_spearman",
                 spearman_correlation,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False)
        return {"val_loss": val_loss}

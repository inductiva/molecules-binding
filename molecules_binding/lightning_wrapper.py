"""
Lightning Code
"""
import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F

spearman = torchmetrics.SpearmanCorrCoef(num_outputs=1)
pearson = torchmetrics.PearsonCorrCoef(num_outputs=1)


class GraphNNLightning(pl.LightningModule):
    """
        Graph Convolution Neural Network
    """

    def __init__(self, model, learning_rate, batch_size, dropout_rate,
                 weight_decay, use_message_passing):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.criterion = torch.nn.MSELoss()
        self.weight_decay = weight_decay
        self.use_message_passing = use_message_passing

    def compute_statistics(self, data, training):
        labels = data.y
        outputs = self.model(data, data.batch, self.dropout_rate,
                             self.use_message_passing).squeeze()

        if training:
            loss = self.criterion(labels, outputs)
            return loss, None, None, None, None

        loss = self.criterion(labels, outputs)
        mae = F.l1_loss(outputs, labels)
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
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

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

    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.weight_decay = weight_decay

    def compute_statistics(self, data, training):
        inputs, labels = data
        inputs = inputs.double()
        labels = labels.double()
        outputs = self.model(inputs).squeeze()

        if training:
            loss = self.criterion(labels, outputs)
            return loss, None, None, None, None

        loss = self.criterion(labels, outputs)
        mae = F.l1_loss(outputs, labels)
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
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

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

"""
Lightning Code
"""
import torch
import pytorch_lightning as pl


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

    def compute_loss(self, data):
        labels = data.y.unsqueeze(1)
        outputs = self.model(data, data.batch, self.dropout_rate)
        return self.criterion(labels, outputs)

    def training_step(self, data, _):
        loss = self.compute_loss(data)
        self.log("loss", loss, batch_size=self.batch_size)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, data, _):
        val_loss = self.compute_loss(data)
        self.log("val_loss",
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}


class MLPLightning(pl.LightningModule):
    """
        Multilayer Perceptron
    """

    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()

    def compute_loss(self, data):
        inputs, labels = data
        inputs = inputs.double()
        labels = torch.unsqueeze(labels, -1).double()
        outputs = self.model(inputs)
        return self.criterion(outputs, labels)

    def training_step(self, data, _):
        loss = self.compute_loss(data)
        self.log("loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, data, _):
        val_loss = self.compute_loss(data)
        self.log("val_loss",
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

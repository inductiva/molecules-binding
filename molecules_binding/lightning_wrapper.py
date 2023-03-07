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

    def training_step(self, data, _):
        labels = data.y.unsqueeze(1)
        outputs = self.model(data, data.batch, self.dropout_rate)
        loss = self.criterion(labels, outputs)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, data, _):
        labels = data.y.unsqueeze(1)
        outputs = self.model(data, data.batch, self.dropout_rate)
        loss = self.criterion(labels, outputs)
        return {"val_loss": loss}

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

    def training_step(self, data, _):
        inputs, labels = data
        inputs = inputs.double()
        labels = torch.unsqueeze(labels, -1).double()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, data, _):
        inputs, labels = data
        inputs = inputs.double()
        labels = torch.unsqueeze(labels, -1).double()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

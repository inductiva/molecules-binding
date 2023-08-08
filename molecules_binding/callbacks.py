""" Callbacks for training"""
from pytorch_lightning import callbacks
import mlflow


class LossMonitor(callbacks.Callback):
    """Logs loss and model checkpoints to mlflow."""

    def __init__(self, run_id):
        self.run_id = run_id

    def on_train_epoch_end(self, trainer, _):
        validation_loss = trainer.callback_metrics['val_loss']
        training_loss = trainer.callback_metrics['loss']

        metrics = {
            'metrics_validation_loss': validation_loss.item(),
            'metrics_training_loss': training_loss.item()
        }

        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics, step=trainer.current_epoch)


class MetricsMonitor(callbacks.Callback):
    """Logs metrics to mlflow."""

    def __init__(self, run_id):
        self.run_id = run_id

    def on_validation_epoch_end(self, trainer, _):
        validation_loss = trainer.callback_metrics['val_loss']
        validation_mae = trainer.callback_metrics['val_mae']
        validation_rmse = trainer.callback_metrics['val_rmse']
        validation_pearson = trainer.callback_metrics['val_pearson']
        validation_spearman = trainer.callback_metrics['val_spearman']

        metrics = {
            'metrics_validation_loss': validation_loss.item(),
            'metrics_validation_mae': validation_mae.item(),
            'metrics_validation_rmse': validation_rmse.item(),
            'metrics_validation_pearson': validation_pearson.item(),
            'metrics_validation_spearman': validation_spearman.item()
        }

        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics, step=trainer.current_epoch)

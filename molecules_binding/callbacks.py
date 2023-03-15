""" Callbacks for training"""
from pytorch_lightning.callbacks import Callback
import mlflow


class LossMonitor(Callback):
    """Logs loss and model checkpoints to mlflow."""

    def __init__(self, run_id):
        self.run_id = run_id

    def on_train_epoch_end(self, trainer, _):
        # Getting the metrics.
        validation_loss = trainer.callback_metrics['val_loss']
        training_loss = trainer.callback_metrics['loss']
        metrics = {
            'metrics_validation_loss': validation_loss.item(),
            'metrics_training_loss': training_loss.item()
        }

        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics, step=trainer.current_epoch)

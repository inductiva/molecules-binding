""" Callbacks for training"""
from pytorch_lightning import callbacks
import mlflow
import os
import logging


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


class MlflowBestModelsCheckpoint(callbacks.Callback):
    """Saves the best models to MLflow according to `monitor_metrics`."""

    def __init__(self, run_id, save_dir, monitor_metrics):
        """
        Args:
          run_id: The MLflow run id.
          save_dir: The directory where the model checkpoints are
             saved before being logged to mlflow.
          monitor_metrics: A list of tuples (validation_loss, mode) where
             validation_loss is the name of the metric to monitor and mode
             is either 'min' or 'max' depending on whether the metric
             should be minimized or maximized respectively.
        """
        self.run_id = run_id
        self.save_dir = save_dir
        # Contruct a dict mapping the monitor metrics to the best
        # value.
        self.best_values = {
            validation_loss: float('inf') if mode == 'min' else float('-inf')
            for validation_loss, mode in monitor_metrics
        }
        self.modes = dict(monitor_metrics)

    def on_validation_end(self, trainer, _):
        for metric in self.best_values:
            self._log_model(trainer, metric)

    def _log_model(self, trainer, metric):
        if metric in trainer.callback_metrics:
            value = trainer.callback_metrics[metric]
            mode = self.modes[metric]
            best_value = self.best_values[metric]
            if (mode == 'min' and value < best_value) or (mode == 'max' and
                                                          value > best_value):
                filename = f'best_{metric}_model_{self.run_id}.ckpt'
                save_path = os.path.join(self.save_dir, filename)
                trainer.save_checkpoint(save_path)
                print('will try to log ', value, ' now. current best value ',
                      self.best_values[metric])
                with mlflow.start_run(run_id=self.run_id):
                    try:
                        mlflow.log_artifact(save_path,
                                            artifact_path='checkpoints')
                        self.best_values[metric] = value
                    # pylint: disable=broad-exception-caught
                    except Exception:
                        print('couldnt log artifact')
        else:
            logging.info(
                'Tried to log metric %s not present in callback_metrics.',
                metric)

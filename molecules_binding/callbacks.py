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

        validation_mae = trainer.callback_metrics['val_mae']
        validation_rmse = trainer.callback_metrics['val_rmse']
        validation_pearson = trainer.callback_metrics['val_pearson']
        validation_spearman = trainer.callback_metrics['val_spearman']

        metrics = {
            'metrics_validation_loss': validation_loss.item(),
            'metrics_training_loss': training_loss.item(),
            'metrics_validation_mae': validation_mae.item(),
            'metrics_validation_rmse': validation_rmse.item(),
            'metrics_validation_pearson': validation_pearson.item(),
            'metrics_validation_spearman': validation_spearman.item()
        }

        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics, step=trainer.current_epoch)


# class Metrics(Callback):

#     def __init__(self, run_id, data) -> None:
#         super().__init__()

#     def on_validation_epoch_end(self, trainer, pl_module):
#         model = pl_module.model

#         pearson_coorrelation = torch.nn.functional.pearsonr(outputs, labels)
#         spearman_correlation = torch.nn.functional.spearmanr(outputs, labels)
#         mae = torch.nn.functional.l1_loss(outputs, labels)
#         rmse = torch.sqrt(torch.nn.functional.mse_loss(outputs, labels))

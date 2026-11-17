from __future__ import annotations

from lightning.pytorch.callbacks import Callback


class BestMetricsAtBestValLoss(Callback):
    """Record train/validation accuracy at the epoch with minimum val_loss."""

    def __init__(self) -> None:
        super().__init__()
        self.best_val_loss = float("inf")
        self.best_train_acc: float | None = None
        self.best_val_acc: float | None = None
        self.best_epoch: int | None = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        train_acc = metrics.get("train_acc")
        val_acc = metrics.get("val_acc")

        if val_loss is None:
            return

        val_loss_float = float(val_loss.detach().cpu())
        if val_loss_float < self.best_val_loss:
            self.best_val_loss = val_loss_float
            self.best_train_acc = (
                float(train_acc.detach().cpu())
                if train_acc is not None
                else None
            )
            self.best_val_acc = (
                float(val_acc.detach().cpu()) if val_acc is not None else None
            )
            self.best_epoch = int(trainer.current_epoch)

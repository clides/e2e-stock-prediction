from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from lstmPredictor import logger
from lstmPredictor.config.configuration import (
    DataIngestionConfigurationManager,
    TrainingConfig,
)
from lstmPredictor.utils.common import save_ptmodel


class LSTMTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.lr_patience,
            factor=config.lr_factor,
        )

        self.device = self._get_device()

        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        self.data_config = DataIngestionConfigurationManager(
            Path(__file__).parent.parent.parent.parent / "params.yaml"
        ).get_data_ingestion_config()

    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_optimizer(self) -> Optimizer:
        """Creates optimizer based on the training config"""
        optimizer_map = {
            "adam": Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        return optimizer_map[self.config.optimizer.lower()](
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

    def _train_epoch(self) -> float:
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            # 1. Forward pass
            preds = self.model(X_batch)

            # 2. Calculate loss
            loss = self.loss_fn(preds, y_batch)

            # 3. Zero grad of optimizer
            self.optimizer.zero_grad()

            # 4. Back propagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            # 5. Gradient descent
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)

                loss = self.loss_fn(outputs, y_batch)
                total_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        # Calculate additional metrics
        avg_loss = total_loss / len(loader)

        return avg_loss

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= self.config.early_stopping_patience

    def _save_checkpoint(
        self, epoch: int, is_best: bool = False, model_name: Optional[str] = None
    ) -> Path:
        """Save model to a checkpoint directory"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": self.best_val_loss,
        }

        save_path = Path(self.config.checkpoint_dir) / self.data_config.ticker
        if is_best:
            name = model_name + "_best_model.pth" if model_name else "best_model.pth"
            save_ptmodel(checkpoint, save_path, name)
            return save_path / name
        if epoch % self.config.checkpoint_freq == 0:
            name = (
                model_name + f"checkpoint_epoch_{epoch}.pth"
                if model_name
                else f"checkpoint_epoch_{epoch}.pth"
            )
            save_ptmodel(checkpoint, save_path, name)
            return save_path / name

    def train(self) -> Path:
        for epoch in range(1, self.config.epochs + 1):
            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss = self._validate_epoch(self.val_loader)
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            model_name = (
                datetime.now() - timedelta(days=self.data_config.num_days)
            ).strftime("%Y-%m-%d")
            if val_loss < self.best_val_loss:
                best_model_path = self._save_checkpoint(
                    epoch, is_best=True, model_name=model_name
                )
            if not best_model_path and epoch % self.config.checkpoint_freq == 0:
                best_model_path = self._save_checkpoint(epoch, model_name=model_name)

            # Early stopping
            if self._check_early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return best_model_path

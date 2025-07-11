from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from lstmPredictor.components.training import LSTMTrainer
from lstmPredictor.config.configuration import TrainingConfigurationManager
from lstmPredictor.constants import PARAMS_FILE_PATH


class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        config_manager = TrainingConfigurationManager(PARAMS_FILE_PATH)
        self.config = config_manager.get_training_config()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def run(self) -> Path:
        trainer = LSTMTrainer(
            config=self.config,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )
        return trainer.train()

from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from lstmPredictor.components.training import LSTMTrainer
from lstmPredictor.config.configuration import TrainingConfigurationManager


class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        config_manager = TrainingConfigurationManager(CONFIG_PATH)
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
        best_model_path = trainer.train()

        return best_model_path

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from lstmPredictor import logger
from lstmPredictor.components.model import StockLSTM
from lstmPredictor.entity.entity import LSTMConfig


class PrepareBaseModel:
    def __init__(self, config: LSTMConfig):
        self.config = config

    def build_lstm(self) -> Tuple[nn.Module, str]:
        """Builds the LSTM architecture from config"""
        try:
            model = StockLSTM(
                self.config.input_size,
                self.config.hidden_size,
                self.config.num_layers,
                self.config.output_size,
                self.config.dropout,
                self.config.bidirectional,
            )

            base_model_path = self._save_model(model, self.config.base_model_path)

            return model, base_model_path
        except Exception as e:
            logger.exception(f"Error building model: {e}")
            raise e

    @staticmethod
    def _save_model(model: StockLSTM, path: Path) -> str:
        """Saves the entire model (architecture + weights)"""
        try:
            os.makedirs(path, exist_ok=True)

            torch.save(model, path / "base_model.pth")
            logger.info(f"Model saved to {path}")

            return str(path / "base_model.pth")
        except Exception as e:
            logger.exception(f"Error saving model: {e}")
            raise e

    @classmethod
    def load_model(cls, path: Path) -> nn.Module:
        """Loads model for prediction pipeline"""
        try:
            model = torch.load(path)
            model.eval()
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            raise e

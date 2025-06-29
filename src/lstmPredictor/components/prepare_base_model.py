import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from lstmPredictor import logger
from lstmPredictor.entity.entity import LSTMConfig


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        drop_out: float,
        bidirectional: bool,
    ):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = drop_out

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_out,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


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

    def _save_model(self, model: StockLSTM, path: Path) -> str:
        """Saves the entire model (architecture + weights)"""
        try:
            os.makedirs(path, exist_ok=True)

            torch.save(model, path / "base_model.pth")
            logger.info(f"Model saved to {path}")

            return str(path / "base_model.pth")
        except Exception as e:
            logger.exception(f"Error saving model: {e}")
            raise e

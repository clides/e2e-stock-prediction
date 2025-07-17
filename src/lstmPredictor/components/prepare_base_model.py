from pathlib import Path

import torch
import torch.nn as nn

from lstmPredictor import logger
from lstmPredictor.entity.entity import LSTMConfig
from lstmPredictor.utils.common import save_ptmodel


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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out


class PrepareBaseModel:
    def __init__(self, config: LSTMConfig):
        self.config = config

    def build_lstm(self) -> Path:
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

            base_model_path = save_ptmodel(
                model=model,
                path=Path(self.config.base_model_path),
                model_name="base_model.pth",
            )

            return base_model_path
        except Exception as e:
            logger.exception(f"Error building model: {e}")
            raise e

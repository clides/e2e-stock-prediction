from pathlib import Path

import torch
import torch.nn as nn

from lstmPredictor import logger
from lstmPredictor.entity.entity import LSTMConfig
from lstmPredictor.utils.common import save_ptmodel, set_random_seed


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
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_out if num_layers > 1 else 0,  # Dropout only between layers
            bidirectional=bidirectional,
        )

        # Enhanced output layer with residual connection
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        # Learnable bias correction term
        self.bias_correction = nn.Parameter(torch.zeros(1))

        # Skip connection to maintain price level
        self.skip = nn.Linear(input_size, output_size)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.2)  # Small positive bias

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        # Initialize hidden states
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
        ).to(device)

        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
        ).to(device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Get last time step's features
        last_features = lstm_out[:, -1, :]

        # Base prediction
        base_pred = self.fc(last_features)

        # Skip connection from last input value
        skip_value = self.skip(x[:, -1, :])

        # Combine with learned bias correction
        out = base_pred + skip_value + self.bias_correction

        return out


class PrepareBaseModel:
    def __init__(self, config: LSTMConfig):
        self.config = config

    def build_lstm(self) -> Path:
        """Builds the LSTM architecture from config"""
        try:
            set_random_seed(self.config.seed)

            model = StockLSTM(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=self.config.output_size,
                drop_out=self.config.dropout,
                bidirectional=self.config.bidirectional,
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

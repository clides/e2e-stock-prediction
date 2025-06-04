from pathlib import Path

from lstmPredictor.entity.entity import LSTMConfig
from lstmPredictor.utils.common import read_yaml


class ConfigurationManager:
    def __init__(self, params_path: Path):
        self.params = read_yaml(params_path)

    def get_base_model_config(self) -> LSTMConfig:
        return LSTMConfig(
            input_size=self.params.base_model.input_size,
            hidden_size=self.params.base_model.hidden_size,
            num_layers=self.params.base_model.num_layers,
            output_size=1,
            dropout=self.params.base_model.dropout,
            bidirectional=self.params.base_model.bidirectional,
            base_model_path=Path("artifacts/models"),
        )

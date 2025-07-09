from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from lstmPredictor.utils.common import validate_date, validate_ticker


@dataclass
class DataIngestionConfig:
    ticker: str
    num_days: int
    raw_data_dir: str

    def __post_init__(self):
        self.ticker = validate_ticker(self.ticker)
        self.start_date = validate_date(self.start_date)

    @property
    def csv_path(self):
        return f"{self.raw_data_dir}/{self.ticker}/{self.start_date}_to_{datetime.now().strftime('%Y-%m-%d')}.csv"


@dataclass
class LSTMConfig:
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    base_model_path: Path


@dataclass
class DataPreprocessingConfig:
    processed_file_path: str
    sequence_length: int
    train_size: float
    test_size: float
    val_size: float
    batch_size: int
    features: list
    target: str
    normalize: bool
    fill_method: str


@dataclass
class TrainingConfig:
    learning_rate: int
    weight_decay: int
    epochs: int
    optimizer: str
    lr_patience: int
    lr_factor: float
    gradient_clip: float
    early_stopping_patience: int
    checkpoint_dir: str
    checkpoint_freq: int


@dataclass
class EvaluationConfig:
    metrics: Dict[str, bool]

from dataclasses import dataclass
from pathlib import Path

from lstmPredictor.utils.common import validate_date, validate_ticker


@dataclass
class DataIngestionConfig:
    ticker: str
    start_date: str
    end_date: str
    raw_data_dir: str = "artifacts/data_ingestion"

    def __post_init__(self):
        self.ticker = validate_ticker(self.ticker)
        self.start_date, self.end_date = validate_date(self.start_date, self.end_date)

    @property
    def csv_path(self):
        return f"{self.raw_data_dir}/{self.ticker}/{self.start_date}_to_{self.end_date}.csv"


@dataclass
class LSTMConfig:
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    base_model_path: Path

from pathlib import Path

from lstmPredictor.entity.entity import DataPreprocessingConfig, LSTMConfig
from lstmPredictor.utils.common import read_yaml


class BaseModelConfigurationManager:
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


class DataPreprocessingConfigurationManager:
    def __init__(self, params_path: Path):
        self.params = read_yaml(params_path)

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        return DataPreprocessingConfig(
            processed_file_path=self.params.data_preprocessing.processed_file_path,
            sequence_length=self.params.data_preprocessing.sequence_length,
            train_size=self.params.data_preprocessing.train_size,
            test_size=self.params.data_preprocessing.test_size,
            val_size=self.params.data_preprocessing.val_size,
            batch_size=self.params.data_preprocessing.batch_size,
            features=self.params.data_preprocessing.features,
            target=self.params.data_preprocessing.target,
            normalize=self.params.data_preprocessing.normalize,
            fill_method=self.params.data_preprocessing.fill_method,
        )

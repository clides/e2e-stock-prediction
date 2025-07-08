from pathlib import Path

from lstmPredictor.components.data_processing import DataPreprocessing
from lstmPredictor.config.configuration import DataPreprocessingConfigurationManager


class DataPreprocessingPipeline:
    def __init__(self, raw_file_path: str):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        config_manager = DataPreprocessingConfigurationManager(CONFIG_PATH)
        self.config = config_manager.get_data_preprocessing_config()
        self.raw_file_path = raw_file_path

    def run(self):
        data_preprocessing_manager = DataPreprocessing(
            config=self.config, raw_file_path=self.raw_file_path
        )
        dataloaders, scaler = data_preprocessing_manager.create_data_loaders()

        return dataloaders, scaler

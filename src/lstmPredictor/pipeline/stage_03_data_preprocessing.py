from pathlib import Path

from lstmPredictor.components.data_processing import DataPreprocessing
from lstmPredictor.config.configuration import DataPreprocessingConfigurationManager


class DataPreprocessingPipeline:
    def __init__(self, raw_file_path: str):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        self.config = DataPreprocessingConfigurationManager(CONFIG_PATH)
        self.raw_file_path = raw_file_path

    def run(self):
        preprocessing_config = self.config.get_data_preprocessing_config()
        data_preprocessing_manager = DataPreprocessing(
            config=preprocessing_config, raw_file_path=self.raw_file_path
        )
        dataloaders = data_preprocessing_manager.create_data_loaders()

        return dataloaders

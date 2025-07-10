from lstmPredictor.components.data_processing import DataPreprocessing
from lstmPredictor.config.configuration import DataPreprocessingConfigurationManager
from lstmPredictor.constants import PARAMS_FILE_PATH


class DataPreprocessingPipeline:
    def __init__(self, raw_file_path: str):
        config_manager = DataPreprocessingConfigurationManager(PARAMS_FILE_PATH)
        self.config = config_manager.get_data_preprocessing_config()
        self.raw_file_path = raw_file_path

    def run(self):
        data_preprocessing_manager = DataPreprocessing(
            config=self.config, raw_file_path=self.raw_file_path
        )
        dataloaders, scaler = data_preprocessing_manager.create_data_loaders()

        return dataloaders, scaler

from lstmPredictor.components.prepare_base_model import PrepareBaseModel
from lstmPredictor.config.configuration import BaseModelConfigurationManager
from lstmPredictor.constants import PARAMS_FILE_PATH


class BaseModelPipeline:
    def __init__(self):
        config_manager = BaseModelConfigurationManager(PARAMS_FILE_PATH)
        self.config = config_manager.get_base_model_config()

    def run(self):
        """
        Execute the complete base model initialization process.

        Returns:
            LSTM model ready for training
        """
        preparer = PrepareBaseModel(self.config)
        base_model_path = preparer.build_lstm()

        return base_model_path

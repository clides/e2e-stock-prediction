from pathlib import Path

from lstmPredictor.components.prepare_base_model import PrepareBaseModel
from lstmPredictor.config.configuration import BaseModelConfigurationManager


class BaseModelPipeline:
    def __init__(self):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        config_manager = BaseModelConfigurationManager(CONFIG_PATH)
        self.config = config_manager.get_base_model_config()

    def run(self):
        """
        Execute the complete base model initialization process.

        Returns:
            LSTM model ready for training
        """
        preparer = PrepareBaseModel(self.config)
        base_model, base_model_path = preparer.build_lstm()

        return base_model, base_model_path

from pathlib import Path

from lstmPredictor.components.prepare_base_model import PrepareBaseModel
from lstmPredictor.config.configuration import BaseModelConfigurationManager


class BaseModelPipeline:
    def __init__(self):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        self.config = BaseModelConfigurationManager(CONFIG_PATH)

    def run(self):
        """
        Execute the complete base model initialization process.

        Returns:
            LSTM model ready for training
        """

        base_model_config = self.config.get_base_model_config()
        preparer = PrepareBaseModel(base_model_config)
        base_model, base_model_path = preparer.build_lstm()

        return base_model, base_model_path

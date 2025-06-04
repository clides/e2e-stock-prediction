from pathlib import Path

from lstmPredictor.components.prepare_base_model import PrepareBaseModel
from lstmPredictor.config.configuration import ConfigurationManager


class BaseModelPipeline:
    def __init__(self):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        self.config = ConfigurationManager(CONFIG_PATH)

    def run(self):
        """
        Execute the complete base model initialization process.

        Returns:
            LSTM model ready for training
        """
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

        base_model_config = self.config.get_base_model_config()
        preparer = PrepareBaseModel(base_model_config)
        model = preparer.build_lstm()

        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")

        return model

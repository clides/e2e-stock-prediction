from pathlib import Path

from lstmPredictor.components.data_ingestion import StockDataIngestion
from lstmPredictor.config.configuration import DataIngestionConfigurationManager


class DataIngestionTrainingPipeline:
    def __init__(
        self,
    ):
        CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"
        config_manager = DataIngestionConfigurationManager(CONFIG_PATH)
        self.config = config_manager.get_data_ingestion_config()

    def run(self) -> str:
        """
        Execute the complete data ingestion process.

        Returns:
            Path where the data was saved
        """
        data_ingestion = StockDataIngestion(self.config)
        saved_path = data_ingestion.run_ingestion()

        return saved_path

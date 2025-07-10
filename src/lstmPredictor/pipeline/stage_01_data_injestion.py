from lstmPredictor.components.data_ingestion import StockDataIngestion
from lstmPredictor.config.configuration import DataIngestionConfigurationManager
from lstmPredictor.constants import PARAMS_FILE_PATH


class DataIngestionPipeline:
    def __init__(self):
        config_manager = DataIngestionConfigurationManager(PARAMS_FILE_PATH)
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

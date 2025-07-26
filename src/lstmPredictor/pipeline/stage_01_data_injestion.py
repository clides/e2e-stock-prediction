from typing import Optional

from lstmPredictor.components.data_ingestion import StockDataIngestion
from lstmPredictor.config.configuration import DataIngestionConfigurationManager
from lstmPredictor.constants import PARAMS_FILE_PATH


class DataIngestionPipeline:
    def __init__(self, ticker: Optional[str] = None, num_days: Optional[int] = None):
        config_manager = DataIngestionConfigurationManager(PARAMS_FILE_PATH)
        self.ticker = ticker
        self.num_days = num_days
        self.config = config_manager.get_data_ingestion_config()

    def run(self) -> str:
        """
        Execute the complete data ingestion process.

        Returns:
            Path where the data was saved
        """
        data_ingestion = StockDataIngestion(
            config=self.config, ticker=self.ticker, num_days=self.num_days
        )
        saved_path = data_ingestion.run_ingestion()

        return saved_path

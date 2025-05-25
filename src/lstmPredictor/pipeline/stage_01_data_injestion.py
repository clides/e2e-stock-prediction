from lstmPredictor.entity.entity import DataIngestionConfig
from lstmPredictor.components.data_ingestion import StockDataIngestion
from lstmPredictor import logger

STAGE_NAME = "(1) Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        Initialize the data ingestion pipeline with user-provided parameters.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    
    def run(self) -> str:
        """
        Execute the complete data ingestion process.
        
        Returns:
            Path where the data was saved
        """
        try:
            logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
            
            # 1. Create and validate configuration
            data_entity = DataIngestionConfig(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # 2. Initialize and run ingestion
            data_ingestion = StockDataIngestion(config=data_entity)
            saved_path = data_ingestion.run_ingestion()
            
            logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
            logger.info(f"Data saved at: {saved_path}")
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {str(e)}")
            raise
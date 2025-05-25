from lstmPredictor import logger
from lstmPredictor.pipeline.stage_01_data_injestion import DataIngestionTrainingPipeline

# default placeholder values for now, will be updated to dynamic info retrieved from user via fastapi later
default_ticker = "AAPL"
default_start_date = "2024-01-01"
default_end_date = "2024-01-31"

try:
    data_ingestion = DataIngestionTrainingPipeline(
        ticker=default_ticker,
        start_date=default_start_date,
        end_date=default_end_date
    )
    data_ingestion.run()
except Exception as e:
    raise e
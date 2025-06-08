from lstmPredictor import logger
from lstmPredictor.pipeline.stage_01_data_injestion import DataIngestionTrainingPipeline
from lstmPredictor.pipeline.stage_02_base_model import BaseModelPipeline

# Stage 1: Data Ingestion
try:
    logger.info(">>>>>> (1) Data Ingestion Stage started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_path = data_ingestion.run()
    logger.info(f"Data saved at: {data_path}")
    logger.info(">>>>>> (1) Data Ingestion Stage coimpleted <<<<<<")
except Exception as e:
    logger.error(f"Error in (1) Data Ingestion Stage: {str(e)}")
    raise e

# Stage 2: Initialize Base Model
try:
    logger.info(">>>>>> (2) Initialize Base Model Stage started <<<<<<")
    base_model_pipeline = BaseModelPipeline()
    base_model, base_model_path = base_model_pipeline.run()
    logger.info(f"Base model saved at: {base_model_path}")
    logger.info(">>>>>> (2) Initialize Base Model Stage completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in (2) Initialize Base Model Stage started: {str(e)}")
    raise e

from lstmPredictor import logger
from lstmPredictor.pipeline.stage_01_data_injestion import DataIngestionTrainingPipeline
from lstmPredictor.pipeline.stage_02_base_model import BaseModelPipeline
from lstmPredictor.pipeline.stage_03_data_preprocessing import DataPreprocessingPipeline
from lstmPredictor.pipeline.stage_04_training import TrainingPipeline

# Stage 1: Data Ingestion
try:
    logger.info(">>>>>> (1) Data Ingestion Stage started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_path = data_ingestion.run()
    logger.info(f"Data saved at: {data_path}")
    logger.info(">>>>>> (1) Data Ingestion Stage completed <<<<<<")
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
    logger.exception(f"Error in (2) Initialize Base Model Stage: {str(e)}")
    raise e

# Stage 3: Preprocess Raw Data
try:
    logger.info(">>>>>> (3) Preprocess Raw Data Stage started <<<<<<")
    data_preprocessing_pipeline = DataPreprocessingPipeline(data_path)
    dataloaders, scaler = data_preprocessing_pipeline.run()
    logger.info(">>>>>> (3) Preprocess Raw Data Stage completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in (3) Preprocess Raw Data Stage: {str(e)}")
    raise e

# Stage 4: Train Model
try:
    logger.info(">>>>>> (4) Model Training Stage started <<<<<<")
    training_pipeline = TrainingPipeline(
        model=base_model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )
    best_model_path = training_pipeline.run()
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(">>>>>> (4) Model Training Stage completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in (4) Model Training Stage: {str(e)}")
    raise e

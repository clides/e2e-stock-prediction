import torch
from celery import Celery

# Import your pipeline stages
from src.lstmPredictor.pipeline.stage_01_data_injestion import DataIngestionPipeline
from src.lstmPredictor.pipeline.stage_02_base_model import BaseModelPipeline
from src.lstmPredictor.pipeline.stage_03_data_preprocessing import (
    DataPreprocessingPipeline,
)
from src.lstmPredictor.pipeline.stage_04_training import TrainingPipeline

# Configure Celery
# The first argument is the name of the current module.
# The `broker` is the URL to our RabbitMQ instance.
# The `backend` is used to store task results. 'rpc://' is a simple backend
# that sends results back over the same AMQP connection.
celery_app = Celery(
    "tasks", broker="amqp://guest:guest@localhost:5672//", backend="rpc://"
)


@celery_app.task
def train_model_task(ticker: str, num_days: int):
    """
    This is the Celery task that runs the entire pipeline.
    It runs in the background, completely separate from the API server.
    """
    try:
        print(f"WORKER: Starting data ingestion for {ticker}...")
        ingestion_pipeline = DataIngestionPipeline(ticker=ticker, num_days=num_days)
        raw_data_path = ingestion_pipeline.run()
        print(f"WORKER: Data ingestion complete. Path: {raw_data_path}")

        print("WORKER: Preparing base model...")
        base_model_pipeline = BaseModelPipeline()
        base_model_path = base_model_pipeline.run()
        print(f"WORKER: Base model prepared. Path: {base_model_path}")

        print("WORKER: Preprocessing data...")
        preprocess_pipeline = DataPreprocessingPipeline(raw_file_path=raw_data_path)
        dataloaders, _ = preprocess_pipeline.run()
        print("WORKER: Data preprocessing complete.")

        print("WORKER: Starting model training...")
        model = torch.load(base_model_path)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        training_pipeline = TrainingPipeline(
            model=model, train_loader=train_loader, val_loader=val_loader
        )
        trained_model_path = training_pipeline.run()
        print(f"WORKER: Training complete. Model saved to: {trained_model_path}")

        # The return value of the task is its result
        return {"status": "SUCCESS", "trained_model_path": str(trained_model_path)}
    except Exception as e:
        # Proper error handling
        print(f"WORKER: Task failed with error: {e}")
        return {"status": "FAILURE", "error": str(e)}

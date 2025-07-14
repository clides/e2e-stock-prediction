import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import torch
import yaml
import yfinance as yf
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from sklearn.preprocessing import MinMaxScaler

from lstmPredictor import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_direcories: list, verbose=True):
    for path in path_to_direcories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def save_bin(path: Path, data: Any):
    joblib.dump(value=data, filename=path)
    logger.info(f"bin file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"bin file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> int:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return size_in_kb


@ensure_annotations
def validate_ticker(ticker: str):
    """Returns the validated ticker if valid, otherwise raises ValueError."""
    ticker = str(ticker).strip().upper()

    # Basic validation
    if not ticker.isalpha():
        raise ValueError(f"Invalid ticker format: {ticker}. Must be alphabetic.")

    # Check if ticker exists
    try:
        test_data = yf.download(ticker, period="1d", progress=False)
        if test_data.empty:
            raise ValueError(
                f"No data found for ticker: {ticker}. Please check the ticker symbol."
            )
    except Exception as e:
        raise ValueError(f"Failed to validate ticker {ticker}: {str(e)}")


@ensure_annotations
def validate_date(start_date: str):
    """Validate date format and logic, returns validated start_date."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        current_date = datetime.now()

        if start >= current_date - timedelta(days=365):
            raise ValueError("start_date must be before end_date")
    except ValueError as e:
        raise ValueError(f"Invalid date format or logic: {e}. Use YYYY-MM-DD.")


@ensure_annotations
def get_default_date_range(days_back: int = 365) -> tuple[str, str]:
    """Get start/end dates (e.g., for API defaults)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_device() -> torch.device:
    """Determine the best available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ensure_annotations
def save_ptmodel(
    model: Union[torch.nn.Module, dict], path: Path, model_name: str
) -> Path:
    """Saves the entire model (architecture + weights)"""
    try:
        os.makedirs(path, exist_ok=True)
        save_path = path / model_name

        torch.save(model, save_path)
        logger.info(f"Model saved to {save_path}")

        return save_path
    except Exception as e:
        logger.exception(f"Error saving model: {e}")
        raise e


@ensure_annotations
def load_ptmodel(path: Path) -> torch.nn.Module:
    """Load a PyTorch model from a given path."""
    try:
        model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise e

    @ensure_annotations
    def inverse_transform_predictions(
        y_pred: np.ndarray, scaler: MinMaxScaler
    ) -> np.ndarray:
        from lstmPredictor.config.configuration import (
            DataPreprocessingConfigurateionManager,
        )

        data_config = DataPreprocessingConfigurateionManager(
            Path(__file__).parent.parent.parent.parent / "params.yaml"
        ).get_data_preprocessing_config()
        target_col_idx = data_config.features.index(data_config.target)

        y_pred = y_pred.reshape(-1, 1)
        dummy_y_pred = np.zeros((y_pred.shape[0], scaler.n_features_in_))

        dummy_y_pred[:, target_col_idx] = y_pred.squeeze()

        inversed_y_pred = scaler.inverse_transform(dummy_y_pred)
        return inversed_y_pred[:, target_col_idx]

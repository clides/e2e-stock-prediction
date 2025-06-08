import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import yaml
import yfinance as yf
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

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
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return size_in_kb


def validate_ticker(ticker: str) -> str:
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
        return ticker
    except Exception as e:
        raise ValueError(f"Failed to validate ticker {ticker}: {str(e)}")


def validate_date(start_date: str) -> str:
    """Validate date format and logic, returns validated start_date."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        current_date = datetime.now()

        if start >= current_date - timedelta(days=365):
            raise ValueError("start_date must be before end_date")
        return start_date
    except ValueError as e:
        raise ValueError(f"Invalid date format or logic: {e}. Use YYYY-MM-DD.")


@ensure_annotations
def get_default_date_range(days_back: int = 365) -> tuple[str, str]:
    """Get start/end dates (e.g., for API defaults)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

import os
from box.exceptions import BoxValueError
import yaml
from lstmPredictor import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import HTTPException
from datetime import datetime, timedelta

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
    size_in_kb = round(os.path.getsize(path)/1024)
    return size_in_kb

@ensure_annotations
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(company_name, start=start_date, end=end_date)
    return data

@ensure_annotations
def validate_ticker(ticker: str) -> bool:
    try:
        data = yf.Ticker(ticker)
        info = data.info
        return info.get("regularMarketPrice") is not None
    except Exception as e:
        return False
    
@ensure_annotations
def get_default_date_range(days_back: int = 365) -> tuple[str, str]:
    """Get start/end dates (e.g., for API defaults)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
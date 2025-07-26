import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from lstmPredictor.entity.entity import DataIngestionConfig
from lstmPredictor.utils.common import validate_date, validate_ticker


class StockDataIngestion:
    def __init__(self, config: DataIngestionConfig, **kwargs: Any):
        """
        Initialize with a validated DataIngestionConfig object.
        """
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        if kwargs.get("ticker", None):
            self.ticker = kwargs["ticker"]
        else:
            self.ticker = config.ticker
        if kwargs.get("num_days", None):
            num_days = kwargs["num_days"]
        else:
            num_days = config.num_days
        self.start_date = datetime.now() - timedelta(days=num_days)

        validate_date(self.start_date.strftime("%Y-%m-%d"))
        validate_ticker(self.ticker)

    def _csv_path(self, raw_data_dir: str, ticker: str, start_date: str) -> str:
        return f"{self.raw_data_dir}/{self.ticker}/{self.start_date}_to_{datetime.now().strftime('%Y-%m-%d')}.csv"

    def _fetch_data(self) -> pd.DataFrame:
        """Download stock data from Yahoo Finance using validated config."""
        try:
            data = yf.download(
                tickers=self.ticker,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=datetime.now().strftime("%Y-%m-%d"),
            )
            if data.empty:
                raise ValueError(
                    f"No data found for {self.config.ticker} in the given range."
                )
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {str(e)}")

    def _save_data(self, data: pd.DataFrame) -> str:
        """Store the raw stock data with only Date, Open, High, Low, Close, Volume columns."""
        try:
            data.columns = data.columns.droplevel("Ticker")
            data.columns.name = None

            cols = ["Open", "High", "Low", "Close", "Volume"]
            data = data[cols]

            csv_path = self._csv_path(
                raw_data_dir=self.raw_data_dir,
                ticker=self.ticker,
                start_date=self.start_date.strftime("%Y-%m-%d"),
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            data.to_csv(csv_path)

            print(f"Successfully saved {len(data)} records to {csv_path}")
            return csv_path
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {str(e)}")

    def run_ingestion(self) -> str:
        """
        Run the complete ingestion process.
        Returns the path where data was saved.
        """
        data = self._fetch_data()
        return self._save_data(data)

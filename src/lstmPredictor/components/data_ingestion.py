import os
from datetime import datetime

import pandas as pd
import yfinance as yf

from lstmPredictor.entity.entity import DataIngestionConfig


class StockDataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize with a validated DataIngestionConfig object.
        """
        self.config = config
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.csv_path), exist_ok=True)

    def _fetch_data(self) -> pd.DataFrame:
        """Download stock data from Yahoo Finance using validated config."""
        try:
            data = yf.download(
                self.config.ticker,
                start=self.config.start_date,
                end=datetime.now().strftime("%Y-%m-%d"),
            )
            if data.empty:
                raise ValueError(
                    f"No data found for {self.config.ticker} in the given range."
                )
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {str(e)}")

    def _save_data(self, data: pd.DataFrame) -> None:
        """Store the raw stock data with only Date, Open, High, Low, Close, Volume columns."""
        try:
            data.columns = data.columns.droplevel("Ticker")
            data.columns.name = None

            cols = ["Open", "High", "Low", "Close", "Volume"]
            data = data[cols]

            data.to_csv(self.config.csv_path)

            print(f"Successfully saved {len(data)} records to {self.config.csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {str(e)}")

    def run_ingestion(self) -> str:
        """
        Run the complete ingestion process.
        Returns the path where data was saved.
        """
        data = self._fetch_data()
        self._save_data(data)
        return self.config.csv_path

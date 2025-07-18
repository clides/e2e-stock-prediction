from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from lstmPredictor import logger
from lstmPredictor.entity.entity import DataPreprocessingConfig
from lstmPredictor.utils.common import set_random_seed


class StockDataset(Dataset):
    """PyTorch Dataset class for stock sequences"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(targets).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig, raw_file_path: str):
        self.config = config
        self.raw_file_path = raw_file_path

    def _load_and_clean_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.raw_file_path)

            if self.config.fill_method == "ffill":
                df.ffill(inplace=True)
            else:
                df.interpolate(inplace=True)

            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _create_sequences(
        self, data: np.ndarray, target_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        target_data = data[:, target_idx]
        sequence_length = self.config.sequence_length

        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(target_data[i + sequence_length])

        return np.array(X), np.array(y)

    def _time_series_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns dictionary of preprocessed and split data:
        {
            "X_train": np.ndarray,
            "y_train": np.ndarray,
            "X_test": np.ndarray,
            "y_test": np.ndarray,
            "X_val": np.ndarray,
            "y_val": np.ndarray
        }
        """
        total_size = len(X)
        test_size = int(total_size * self.config.test_size)
        val_size = int(total_size * self.config.val_size)
        train_size = total_size - test_size - val_size

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = (
            X[train_size : train_size + val_size],
            y[train_size : train_size + val_size],
        )
        X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

        datasets_dict = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_val": X_val,
            "y_val": y_val,
        }

        return datasets_dict

    def _save_artifacts(self, data_dict: Dict[str, np.ndarray]):
        """Save the preprocessed datasets to numpy files"""
        processed_file_path = self.config.processed_file_path
        Path(processed_file_path).mkdir(exist_ok=True)

        for name, arr in data_dict.items():
            np.save(Path(processed_file_path) / f"{name}.npy", arr)

    def _preprocess(self) -> Tuple[Dict[str, np.ndarray], Optional[MinMaxScaler]]:
        # Load and clean raw data into a DataFrame
        df = self._load_and_clean_data()

        # Feature selection
        features = self.config.features
        df_features = df[features]

        # Time-series split on the DataFrame first
        total_size = len(df_features)
        test_size = int(total_size * self.config.test_size)
        val_size = int(total_size * self.config.val_size)
        train_size = total_size - test_size - val_size

        df_train = df_features.iloc[:train_size]
        df_val = df_features.iloc[train_size : train_size + val_size]
        df_test = df_features.iloc[train_size + val_size :]

        # Normalization (fitting only on training data)
        scaler = None
        if self.config.normalize:
            scaler = MinMaxScaler()
            # Fit on training data and transform all sets
            scaled_train_data = scaler.fit_transform(df_train)
            scaled_val_data = scaler.transform(df_val)
            scaled_test_data = scaler.transform(df_test)
        else:
            scaled_train_data = df_train.values
            scaled_val_data = df_val.values
            scaled_test_data = df_test.values

        target_idx = df_features.columns.get_loc(self.config.target)

        # Create sequences from each scaled dataset
        X_train, y_train = self._create_sequences(scaled_train_data, target_idx)
        X_val, y_val = self._create_sequences(scaled_val_data, target_idx)
        X_test, y_test = self._create_sequences(scaled_test_data, target_idx)

        # Combine into a single dictionary
        split_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

        self._save_artifacts(split_data)
        logger.info(
            f"Data preprocessing completed and artifacts saved to {self.config.processed_file_path}"
        )

        return split_data, scaler

    def create_data_loaders(self) -> Tuple[Dict[str, DataLoader], MinMaxScaler]:
        """Returns dictionary of DataLoaders:
        {
            "X_train": DataLoader,
            "y_train": DataLoader,
            "X_test": DataLoader,
            "y_test": DataLoader,
            "X_val": DataLoader,
            "y_val": DataLoader
        }
        """
        set_random_seed(self.config.seed)
        preprocessed_data, scaler = self._preprocess()
        batch_size = self.config.batch_size

        datasets = {
            "train": StockDataset(
                preprocessed_data["X_train"], preprocessed_data["y_train"]
            ),
            "val": StockDataset(preprocessed_data["X_val"], preprocessed_data["y_val"]),
            "test": StockDataset(
                preprocessed_data["X_test"], preprocessed_data["y_test"]
            ),
        }

        loaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=batch_size,
                shuffle=False,
            ),
            "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
        }

        return loaders, scaler

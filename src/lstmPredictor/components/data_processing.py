from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from lstmPredictor import logger
from lstmPredictor.entity.entity import DataPreprocessingConfig


class StockDataset(Dataset):
    """PyTorch Dataset class for stock sequences"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(targets)

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
        feature_data = np.delete(data, target_idx, axis=1)
        target_data = data[:, target_idx]
        sequence_length = self.config.sequence_length

        for i in range(len(data) - sequence_length):
            X.append(feature_data[i : i + sequence_length])
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
        train_split_idx = int(len(X) * (1 - self.config.val_size))
        test_split_idx = train_split_idx + int(len(X) * (1 - self.config.test_size))
        val_split_idx = test_split_idx + int(len(X) * (1 - self.config.val_size))

        X_train, y_train = X[:train_split_idx], y[:train_split_idx]
        X_test, y_test = (
            X[train_split_idx:test_split_idx],
            y[train_split_idx:test_split_idx],
        )
        X_val, y_val = X[test_split_idx:val_split_idx], y[test_split_idx:val_split_idx]

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

    def _preprocess(self) -> Dict[str, np.ndarray]:
        # load and clean raw data into dataframe
        df = self._load_and_clean_data()

        # feature selection
        features = self.config.features
        data = df[features].values

        # normalization
        if self.config.normalize:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

        # create sequences
        X, y = self._create_sequences(
            data=data, target_idx=df.columns.get_loc(self.config.target)
        )

        # split and save data into training, testing, validation datasets
        split_data = self._time_series_split(X, y)
        self._save_artifacts(split_data)
        logger.info(
            f"Data preprocessing completed and saved processed data to numpy files: {self.config.processed_file_path}"
        )

        return split_data

    def create_data_loaders(self) -> Dict[str, DataLoader]:
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
        preprocessed_data = self._preprocess()
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

        return loaders

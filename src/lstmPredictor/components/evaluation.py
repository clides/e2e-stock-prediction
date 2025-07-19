import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader

from lstmPredictor.config.configuration import EvaluationConfig
from lstmPredictor.constants import PARAMS_FILE_PATH
from lstmPredictor.utils.common import (
    get_device,
    inverse_transform_predictions,
    read_yaml,
)


class EvaluateModel:
    def __init__(
        self,
        model: nn.Module,
        test_data: DataLoader,
        scaler: MinMaxScaler,
        config: EvaluationConfig,
    ):
        self.model = model
        self.test_data = test_data
        self.scaler = scaler
        self.config = config
        self.device = get_device()

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_data:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        return y_true, y_pred

    # Normal metric calculation functions
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of Determination (R^2)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))

    # Financial metric calculation functions (higher is better)
    def _calculate_returns(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate returns based on predictions from the model"""
        pred_returns = np.diff(y_pred) / (y_pred[:-1] + 1e-10)
        actual_returns = np.diff(y_true) / (y_true[:-1] + 1e-10)
        return pred_returns, actual_returns

    def _calculate_directional_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Directional Accuracy"""
        pred_returns, actual_returns = self._calculate_returns(y_true, y_pred)
        correct_direction = np.sign(pred_returns) == np.sign(actual_returns)
        return np.mean(correct_direction)

    def _calculate_magnitude_correlation(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Magnitude Correlation"""
        pred_returns, actual_returns = self._calculate_returns(y_true, y_pred)
        return np.corrcoef(pred_returns, actual_returns)[0, 1]

    def _calculate_sharpe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Risk-Adjusted Returns (using confidence weighting)"""
        pred_returns, actual_returns = self._calculate_returns(y_true, y_pred)
        weights = np.tanh(pred_returns * 2)
        strategy_returns = weights * actual_returns
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)
        return sharpe

    def _save_comparision_graph(
        self, y_true: np.ndarray, y_pred: np.ndarray, name: str
    ):
        """Save a comparison graph of true vs predicted values"""
        path = f"artifacts/{name}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="True Values", color="blue")
        plt.plot(y_pred, label="Predicted Values", color="orange")
        plt.title("True vs Predicted Values")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        plt.savefig(path)
        plt.close()

    def _log_model(
        self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float]
    ):
        """Log the model, parameters, and metrics to MLflow"""
        from datetime import datetime

        config = read_yaml(PARAMS_FILE_PATH)
        data_ingestion = dict(config.data_ingestion)
        base_model = dict(config.base_model)
        data_preprocessing = dict(config.data_preprocessing)
        training = dict(config.training)

        params = {}
        random_seed = base_model["seed"]
        del base_model["seed"]
        del data_preprocessing["seed"]
        del training["seed"]
        params.update(data_ingestion)
        params.update(base_model)
        params.update(data_preprocessing)
        params.update(training)
        params["random_seed"] = random_seed

        experiment_name = f"LSTM Stock Price Prediction for {data_ingestion['ticker']}"
        run_name = f"{data_ingestion['ticker']} {datetime.now().strftime('%Y-%m-%d')}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                registered_model_name=f"{data_ingestion['ticker']} Stock Price Predictor (Trained on {datetime.now().strftime('%Y-%m-%d')})",
            )
            if self.config.save_graph:
                graph_name = f"{data_ingestion['ticker']}-evaluation-comparison[{datetime.now().strftime('%Y-%m-%d')}]"
                self._save_comparision_graph(y_true, y_pred, graph_name)
                mlflow.log_artifact(
                    "artifacts/" + graph_name + ".png",
                    artifact_path="evaluation-graphs",
                )

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        scaled_y_true, scaled_y_pred = self._get_predictions()
        y_true = inverse_transform_predictions(np.array(scaled_y_true), self.scaler)
        y_pred = inverse_transform_predictions(np.array(scaled_y_pred), self.scaler)

        selected_metrics = self.config.metrics
        results = {}
        if "MeanAbsoluteError" in selected_metrics:
            results["MeanAbsoluteError"] = self._calculate_mae(y_true, y_pred)
        if "RootMeanSquaredError" in selected_metrics:
            results["RootMeanSquaredError"] = self._calculate_rmse(y_true, y_pred)
        if "R^2" in selected_metrics:
            results["R-Squared"] = self._calculate_r2(y_true, y_pred)
        if "MeanAbsolutePercentageError" in selected_metrics:
            results["MeanAbsolutePercentageError"] = self._calculate_mape(
                y_true, y_pred
            )
        if "DirectionalAccuracy" in selected_metrics:
            results["DirectionalAccuracy"] = self._calculate_directional_accuracy(
                y_true, y_pred
            )
        if "MagCorr" in selected_metrics:
            results["MagnitudeCorrelation"] = self._calculate_magnitude_correlation(
                y_true, y_pred
            )
        if "SharpeRatio" in selected_metrics:
            results["SharpeRatio"] = self._calculate_sharpe(y_true, y_pred)

        if self.config.log_model:
            self._log_model(y_true, y_pred, results)

        return results

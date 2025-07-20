from datetime import date, timedelta

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from lstmPredictor import logger
from lstmPredictor.utils.common import inverse_transform_predictions


class MakePredictions:
    def __init__(self, model: nn.Module, scaler: MinMaxScaler):
        self.model = model
        self.scaler = scaler

        X_test = np.load("artifacts/data/processed_data/X_test.npy")
        data = np.expand_dims(X_test[-1], axis=0)
        self.data = data

    def predict(self) -> float:
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(self.data, dtype=torch.float32)
            raw_prediction = self.model(data_tensor)

        prediction = inverse_transform_predictions(
            y_pred=raw_prediction.numpy(), scaler=self.scaler
        )

        today = date.today()
        next_day = today + timedelta(days=1)
        if next_day.weekday() >= 5:
            days_to_add = 7 - next_day.weekday()
            next_day += timedelta(days=days_to_add)
        next_day = next_day.strftime("%A, %B %d, %Y")

        logger.info(f"Predicted value for {next_day}: {prediction[0]:.2f}")

from sklearn.preprocessing import MinMaxScaler
from torch import nn

from lstmPredictor.components.inference import MakePredictions


class MakePredictionsPipeline:
    def __init__(self, model: nn.Module, scaler: MinMaxScaler):
        self.model = model
        self.scaler = scaler

    def run(self) -> float:
        """
        Execute the prediction process using the provided model.
        Returns:
            Predicted value for the next day
        """
        predictor = MakePredictions(self.model, self.scaler)
        prediction = predictor.predict()
        return prediction

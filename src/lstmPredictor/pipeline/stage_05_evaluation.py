from typing import Dict

from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader

from lstmPredictor.components.evaluation import EvaluateModel
from lstmPredictor.config.configuration import EvaluationConfigurationManager
from lstmPredictor.constants import PARAMS_FILE_PATH


class EvaluationPipeline:
    def __init__(
        self,
        model: nn.Module,
        test_data: DataLoader,
        scaler: MinMaxScaler,
    ):
        self.model = model
        self.test_data = test_data
        self.scaler = scaler

        config_manager = EvaluationConfigurationManager(PARAMS_FILE_PATH)
        self.config = config_manager.get_evaluation_config()

    def run(self) -> Dict[str, float]:
        evaluator = EvaluateModel(
            config=self.config,
            model=self.model,
            test_data=self.test_data,
            scaler=self.scaler,
        )

        return evaluator.evaluate()

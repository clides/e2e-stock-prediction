from torch import nn
from torch.utils.data import DataLoader

from lstmPredictor.config.configuration import EvaluationConfig


class EvaluateModel:
    def __init__(
        self, model: nn.Module, test_data: DataLoader, config: EvaluationConfig
    ):
        self.model = model
        self.test_data = test_data
        self.config = config

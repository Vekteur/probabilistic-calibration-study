import torch

from .base_dist_module import DistModule


class DistIC_Regul(DistModule):
    """
    Implementation of the model from the paper "Individual Calibration with Randomized Forecasting".
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomized_predictions = True
    
    def build_model(self):
        return super().build_model(persistent_input_size=1)
    
    def compute_regul(self, dist, y):
        pit = dist.cdf(y)
        regul = torch.abs(pit - self.r).mean()
        return regul
    
    def predict(self, x):
        self.r = torch.rand(x.shape[0], 1, device=x.device)
        return self.model.dist(x, self.r)

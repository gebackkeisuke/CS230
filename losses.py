
import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_map=None):
        super().__init__()
        self.weight_map = weight_map

    def forward(self, pred, target):
        diff = (pred - target) ** 2
        if self.weight_map is not None:
            weighted_diff = diff * self.weight_map
            loss = torch.mean(weighted_diff)
        else:
            loss = torch.mean(diff)

        return loss.mean()
    
def get_loss(cfg):
    loss_type = cfg.get("type", "MSELoss")

    if loss_type == "MSELoss":
        return nn.MSELoss()
    elif loss_type == "L1Loss":
        return nn.L1Loss()
    elif loss_type == "WeightedMSELoss":
        weight_map = cfg.get("weight_map", None)
        return WeightedMSELoss(weight_map=weight_map)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
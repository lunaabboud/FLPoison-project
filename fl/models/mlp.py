import torch.nn as nn
from fl.models import model_registry


@model_registry
class mlp(nn.Module):
    def __init__(self, input_dim=82, num_classes=12):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.float()
        return self.network(x)

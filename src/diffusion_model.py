import torch
import torch.nn as nn

print("diffusion_model.py is being loaded")

class Simple1DDiffusionModel(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len)
        )

    def forward(self, x):
        return self.net(x)

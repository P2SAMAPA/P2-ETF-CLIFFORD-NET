import torch
import torch.nn as nn
from clifford_layers import CliffordLinear

class MultivectorEncoder(nn.Module):
    """Map 11‑dim feature vector to an initial multivector (16 blades)."""
    def __init__(self, n_blades=16):
        super().__init__()
        self.fc = nn.Linear(11, n_blades)

    def forward(self, x):
        # x: (batch, n_etfs, 11)
        mv = self.fc(x)                     # (batch, n_etfs, 16)
        return mv.unsqueeze(-2)             # (batch, n_etfs, 1, 16)

class CliffordNet(nn.Module):
    def __init__(self, input_mv_dim=1, hidden_mv_dim=8, output_mv_dim=1, n_blades=16):
        super().__init__()
        self.encoder = MultivectorEncoder(n_blades)
        self.fc1 = CliffordLinear(input_mv_dim, hidden_mv_dim, n_blades)
        self.relu = nn.ReLU()
        self.fc2 = CliffordLinear(hidden_mv_dim, output_mv_dim, n_blades)

    def forward(self, x):
        # x: (batch, n_etfs, 11)
        x = self.encoder(x)                # (batch, n_etfs, 1, 16)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)                    # (batch, n_etfs, output_mv_dim, 16)
        scalar = x[..., 0]                 # scalar part (blade 0) -> (batch, n_etfs)
        return scalar

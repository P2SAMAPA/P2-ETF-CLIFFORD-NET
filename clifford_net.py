import torch
import torch.nn as nn
from clifford_layers import CliffordNet as CliffordNetBase

class MultivectorEncoder(nn.Module):
    """Construct multivector from scalar, vector, bivector features."""
    def __init__(self, n_blades=16):
        super().__init__()
        self.n_blades = n_blades
        # We'll create a linear projection to map input features to 16 blades (initial multivector)
        # For each ETF, we have 11 features (1 + 4 + 6). We'll map to 16 dims.
        self.fc = nn.Linear(11, n_blades)

    def forward(self, x):
        # x: (batch, n_etfs, 11)
        mv = self.fc(x)  # (batch, n_etfs, 16)
        # Add an extra dimension for mv dimension (set to 1)
        return mv.unsqueeze(-2)  # (batch, n_etfs, 1, 16)

class CliffordNetWrapper(nn.Module):
    def __init__(self, input_mv_dim=1, hidden_mv_dim=8, output_mv_dim=1, n_blades=16):
        super().__init__()
        self.net = CliffordNetBase(input_mv_dim, hidden_mv_dim, output_mv_dim, n_blades)
        self.encoder = MultivectorEncoder(n_blades)

    def forward(self, x):
        # x: (batch, n_etfs, 11)
        mv = self.encoder(x)          # (batch, n_etfs, 1, 16)
        scalar = self.net(mv)         # (batch, n_etfs)
        return scalar

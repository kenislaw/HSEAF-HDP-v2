import torch
import torch.nn as nn
class CreativeExplorationHead(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.creative_diffuser = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.efficiency_head = nn.Linear(latent_dim, 1)
    def forward(self, latent_z, uncertainty):
        if uncertainty.mean() > 0.3: return torch.zeros_like(latent_z)
        delta = self.creative_diffuser(latent_z)
        eff = torch.sigmoid(self.efficiency_head(delta))
        return delta * eff

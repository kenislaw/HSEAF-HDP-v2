import torch
import torch.nn as nn
import torch.nn.functional as F
class EnergyAwareHead(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.energy_net = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, 1), nn.Softplus())
        self.reward_shaper = nn.Linear(latent_dim, 1)
    def forward(self, latent_z, creative_delta):
        energy = self.energy_net(latent_z)
        shaped = self.reward_shaper(latent_z) - 0.1 * energy
        return shaped, energy
    def energy_loss(self, energy_cost):
        return F.huber_loss(energy_cost, torch.zeros_like(energy_cost))

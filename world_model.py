import torch
import torch.nn as nn
import torch.nn.functional as F
class LatentWorldModel(nn.Module):
    def __init__(self, obs_dim=27, goal_dim=2, action_dim=8, latent_dim=128):
        super().__init__()
        self.encoder = nn.Linear(obs_dim + goal_dim, latent_dim)
        self.dynamics = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4, dim_feedforward=512, activation=F.gelu, norm_first=True, batch_first=True)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.next_latent_head = nn.Linear(latent_dim, latent_dim)
        self.uncertainty_head = nn.Linear(latent_dim, 1)
    def forward(self, obs, goal, action):
        x = torch.cat([obs, goal], dim=-1)
        z = self.encoder(x)
        z_dyn = self.dynamics(z.unsqueeze(1)).squeeze(1)
        next_z = self.next_latent_head(z_dyn) + action
        rew = self.reward_head(z_dyn)
        uncertainty = torch.sigmoid(self.uncertainty_head(z_dyn))
        return next_z, rew, uncertainty

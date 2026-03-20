import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from hierarchical_meta import HierarchicalMetaNet  # Phase 2+

class DiTBackbone(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Linear(state_dim + action_dim, hidden_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, activation=F.gelu, norm_first=True, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.patch_embed(x)
        t_emb = self.time_embed(t.unsqueeze(-1))
        x = x + t_emb.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class DiT1D(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dit = DiTBackbone(state_dim, action_dim, hidden_dim)
        self.meta_net = HierarchicalMetaNet(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, action_dim)
        self.escape_head = nn.Linear(hidden_dim, action_dim)
        self.completion_head = nn.Linear(hidden_dim, 1)
        self.noise_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        feat = self.dit(x, t)
        meta_score, subgoal, switch_prob = self.meta_net(feat)
        vel = self.velocity_head(feat)
        escape = self.escape_head(feat) if meta_score.mean() < 1.05 else torch.zeros_like(vel)
        completion = self.completion_head(feat.mean(dim=1))
        return {'velocity': vel, 'escape': escape, 'completion': completion, 'meta_score': meta_score, 'subgoal': subgoal, 'switch_prob': switch_prob}

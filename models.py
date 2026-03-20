import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from hierarchical_meta import HierarchicalMetaNet  # Phase 2+

class DiT1D(nn.Module):
    """DiT1D: single-step diffusion transformer (no sequence dimension).  
    Input: flat [B, state+action]  
    Output: flat [B, action_dim] for all heads."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Linear(state_dim + action_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Transformer layers treat input as [B, 1, hidden] internally but we flatten after
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                activation=F.gelu,
                norm_first=True,
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Heads operate on flat [B, hidden]
        self.meta_net = HierarchicalMetaNet(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, action_dim)
        self.escape_head = nn.Linear(hidden_dim, action_dim)
        self.completion_head = nn.Linear(hidden_dim, 1)
        self.noise_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor):
        x = torch.cat([state, action], dim=-1)              # [B, state+action]
        x = self.patch_embed(x)                             # [B, hidden]

        t_emb = self.time_embed(t.unsqueeze(-1))            # [B, hidden]
        x = x + t_emb

        # Ensure flat before transformer (no seq dim)
        if x.dim() == 3:  # if accidentally [B, seq, hidden]
            x = x.mean(dim=1)  # collapse seq if exists

        for block in self.blocks:
            # Add dummy seq dim only for transformer
            x_block = x.unsqueeze(1)  # [B, 1, hidden]
            x_block = block(x_block)
            x = x_block.squeeze(1)    # back to [B, hidden]

        x = self.norm(x)  # [B, hidden]

        # Explicitly check shape before meta_net
        if x.shape[1] != self.hidden_dim:
            raise RuntimeError(f"Unexpected feature shape before meta_net: {x.shape}")

        meta_score, subgoal, switch_prob = self.meta_net(x)
        vel = self.velocity_head(x)
        escape = self.escape_head(x) if meta_score.mean() < 1.05 else torch.zeros_like(vel)
        completion = self.completion_head(x)  # [B, 1] - no mean needed

        return {
            'velocity': vel,
            'escape': escape,
            'completion': completion,
            'meta_score': meta_score,
            'subgoal': subgoal,
            'switch_prob': switch_prob
        }

class ValueCritic(nn.Module):
    """Twin Q-critics + target networks with Polyak averaging (τ=0.005).  
    Used in ReinFlow PPO phase for value estimation and advantage computation."""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.net1 = mlp()
        self.net2 = mlp()
        self.target_net1 = mlp()
        self.target_net2 = mlp()

        # Hard copy online → target at init
        for target, online in zip([self.target_net1, self.target_net2], [self.net1, self.net2]):
            target.load_state_dict(online.state_dict())

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return twin Q-values."""
        return self.net1(state), self.net2(state)

    def target_forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return twin target Q-values (for min in TD target)."""
        return self.target_net1(state), self.target_net2(state)

    def update_targets(self, tau: float = 0.005):
        """Polyak averaging: target ← τ * online + (1-τ) * target"""
        for target, online in zip([self.target_net1, self.target_net2], [self.net1, self.net2]):
            for p_t, p_o in zip(target.parameters(), online.parameters()):
                p_t.data.copy_(tau * p_o.data + (1 - tau) * p_t.data)

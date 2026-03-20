import torch
import torch.nn as nn
class HierarchicalMetaNet(nn.Module):
    def __init__(self, hidden_dim=256, goal_dim=2):
        super().__init__()
        self.meta_net = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.subgoal_head = nn.Linear(hidden_dim, goal_dim)
        self.switch_vote = nn.Linear(hidden_dim, 1)
    def forward(self, feat):
        meta = self.meta_net(feat)                # [B, 1]
        subgoal = self.subgoal_head(feat)         # [B, goal_dim]
        switch = torch.sigmoid(self.switch_vote(feat))  # [B, 1]
        return meta, subgoal, switch


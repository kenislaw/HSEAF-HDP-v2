import torch
import gymnasium as gym
import numpy as np
import minari
from tqdm import tqdm
import argparse

from models import DiT1D
from world_model import LatentWorldModel
from hierarchical_meta import HierarchicalMetaNet
from her_buffer import SuccessBuffer
from creative_exploration import CreativeExplorationHead
from energy_aware import EnergyAwareHead
from utils import make_d4rl_score, save_rollout_video

class GoalConditionedDiT(DiT1D):
    def forward(self, state, goal, action, t):
        x = torch.cat([state, goal, action], dim=-1)
        return super().forward(x[:, :self.dit.patch_embed.in_features], action, t)

def mppi_planning(model, critic, world_model, creative_head, energy_head, obs, goal, horizon=50, K=12, lam=1.0):
    device = next(model.parameters()).device
    state = torch.tensor(obs['observation'], dtype=torch.float32, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    best_action = None
    best_return = -float('inf')
    for _ in range(3):
        actions = torch.randn(K, horizon, 8, device=device)
        returns = torch.zeros(K, device=device)
        for k in range(K):
            sim_state = state.clone()
            sim_ret = 0.0
            for h in range(horizon):
                t = torch.rand(1, device=device)
                out = model(sim_state, goal_t, actions[k:k+1, h], t)
                a = out['velocity'] + actions[k:k+1, h]
                if out['meta_score'].item() < 1.05:
                    a += out['escape']
                next_z, rew_pred, uncertainty = world_model(sim_state, goal_t, a)
                shortcut = creative_head(next_z, uncertainty)
                shaped_r, energy = energy_head(next_z, shortcut)
                sim_ret += shaped_r.item() * 2.0 - energy.item()
                dist = torch.norm(sim_state[:, :2] - goal_t[:, :2]).item()
                sim_ret += max(0, 1 - dist)
                sim_state = next_z
            returns[k] = sim_ret
        weights = torch.softmax(returns / lam, dim=0)
        best_k = returns.argmax()
        best_action = actions[best_k, 0] if returns[best_k] > best_return else best_action
        best_return = max(best_return, returns[best_k])
    return best_action.cpu().numpy()

def antmaze_hybrid(env_name: str = "AntMaze_Large-v4", episodes: int = 50):
    env = gym.make(env_name, render_mode="rgb_array")
    dataset = minari.load_dataset("D4RL/antmaze/large-play-v1")
    model = GoalConditionedDiT(27, 8)
    model.load_state_dict(torch.load("hseaf_antmaze_finetune.pt", weights_only=True)["model"])
    critic = ValueCritic(27)
    critic.load_state_dict(torch.load("hseaf_antmaze_finetune.pt", weights_only=True)["critic"])
    world_model = LatentWorldModel()
    creative_head = CreativeExplorationHead()
    energy_head = EnergyAwareHead()
    success_buffer = SuccessBuffer()
    optimizer_distill = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.eval()

    successes = 0
    for ep in tqdm(range(episodes)):
        obs, _ = env.reset()
        goal = obs['desired_goal']
        done = False
        steps = 0
        while not done and steps < 1000:
            action = mppi_planning(model, critic, world_model, creative_head, energy_head, obs, goal)
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if np.linalg.norm(obs['achieved_goal'] - goal) < 0.5:
                successes += 1
                break
        success_buffer.add(...)  # fill with rollout data in real run
        if ep % 10 == 0:
            success_buffer.distill_high_meta(model, optimizer_distill)
    score = (successes / episodes) * 100
    print(f"✅ AntMaze success: {score:.1f}%")
    save_rollout_video(lambda o: mppi_planning(model, critic, world_model, creative_head, energy_head, o, o['desired_goal']), env_name)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="AntMaze_Large-v4")
    args = parser.parse_args()
    antmaze_hybrid(args.env)

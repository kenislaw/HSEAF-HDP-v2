import torch
import torch.nn.functional as F
import minari
import gymnasium as gym
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from models import DiT1D
from utils import make_d4rl_score

class MinariTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.states = []
        self.actions = []
        for ep in dataset:
            self.states.extend(ep["observations"])
            self.actions.extend(ep["actions"])
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def train_dit_offline(env_name: str = "Hopper-v5", epochs: int = 50, batch_size: int = 256, lr: float = 3e-4):
    ds_id = f"mujoco/{env_name.lower().replace('-v5','')}/medium-v0"
    if "antmaze" in env_name.lower():
        ds_id = "D4RL/antmaze/large-play-v1"
    dataset = minari.load_dataset(ds_id)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = DiT1D(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    traj_ds = MinariTrajectoryDataset(dataset)
    loader = DataLoader(traj_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for states, actions in tqdm(loader, desc=f"Epoch {epoch+1}"):
            B = states.shape[0]
            t = torch.rand(B)
            noise = torch.randn_like(actions)
            xt = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * actions
            out = model(states, xt, t)
            target_vel = actions - noise
            vel_loss = F.mse_loss(out["velocity"], target_vel)
            meta_loss = (out["meta_score"] - 1.0).pow(2).mean()
            loss = vel_loss + 0.1 * meta_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), f"hseaf_{env_name.lower()}_offline.pt")
    print(f"✅ Offline pre-training complete ({env_name})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    args = parser.parse_args()
    train_dit_offline(args.env)

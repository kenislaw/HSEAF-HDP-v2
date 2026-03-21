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
        skipped = 0
        for ep in dataset:
            obs = ep.observations
            acts = ep.actions
            min_len = min(len(obs), len(acts))
            if min_len == 0:
                skipped += 1
                continue
            self.states.extend(obs[:min_len])
            self.actions.extend(acts[:min_len])
        if skipped > 0:
            print(f"Warning: skipped {skipped} empty/invalid episodes")
        if len(self.states) == 0:
            raise ValueError("No valid transitions found in dataset")
        if len(self.states) != len(self.actions):
            raise RuntimeError(f"Length mismatch after sync: states={len(self.states)}, actions={len(self.actions)}")
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        print(f"Loaded {len(self.states)} valid transitions")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Safety guard against DataLoader bugs / off-by-one
        if idx >= len(self.states) or idx >= len(self.actions):
            raise IndexError(f"Index {idx} out of bounds (states len: {len(self.states)}, actions len: {len(self.actions)})")
        return self.states[idx], self.actions[idx]

def train_dit_offline(env_name: str = "Hopper-v5", epochs: int = 20, batch_size: int = 256, lr: float = 3e-4):
    ds_id = f"mujoco/{env_name.lower().replace('-v5','')}/medium-v0"
    if "antmaze" in env_name.lower():
        ds_id = "D4RL/antmaze/large-play-v1"
    
    dataset = minari.load_dataset(ds_id, download=True)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = DiT1D(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = f"checkpoints/{env_name.lower()}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_return = -float('inf')

    traj_ds = MinariTrajectoryDataset(dataset)
    loader = DataLoader(traj_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        def quick_eval(model, env, n_ep=5, max_steps=1000):
        model.eval()
        returns = []
        for _ in range(n_ep):
            obs, _ = env.reset()
            ep_ret = 0.0
            for _ in range(max_steps):
                with torch.no_grad():
                    # simple one-step diffusion sample (t=0)
                    t = torch.zeros(1, device=obs.device)
                    noise = torch.randn(action_dim, device=obs.device)
                    xt = noise
                    out = model(obs.unsqueeze(0), xt.unsqueeze(0), t)
                    act = out['velocity'][0].cpu().numpy()
                obs, rew, done, _, _ = env.step(act)
                ep_ret += rew
                if done: break
            returns.append(ep_ret)
        model.train()
        return sum(returns) / len(returns)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for states, actions in tqdm(loader, desc=f"Epoch {epoch+1}"):
            t = torch.rand(len(states), device=states.device)
            noise = torch.randn_like(actions)
            xt = (1 - t[:, None]) * noise + t[:, None] * actions
            
            out = model(states, xt, t)
            
            target_vel = actions - noise
            vel_loss = F.mse_loss(out["velocity"], target_vel)
            meta_loss = (out["meta_score"] - 1.0).pow(2).mean()
            loss = vel_loss + 0.1 * meta_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * states.size(0)

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(traj_ds):.6f}")

            if (epoch + 1) % eval_freq == 0:
            avg_ret = quick_eval(model, env)
            ckpt_path = f"{ckpt_dir}/epoch_{(epoch+1):04d}_ret{avg_ret:.1f}.pt"
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'avg_return': avg_ret
            }, ckpt_path)
            print(f"Checkpoint + metric → {ckpt_path} | AvgReturn: {avg_ret:.2f}")
            if avg_ret > best_return:
                best_return = avg_ret
                torch.save(model.state_dict(), f"hseaf_{env_name.lower()}_offline_best.pt")
                print("  → New best model saved")

    save_path = f"hseaf_{env_name.lower()}_offline.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Offline pre-training complete ({env_name}) → {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    args = parser.parse_args()
    train_dit_offline(args.env)

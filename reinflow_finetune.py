import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import argparse

from models import DiT1D, ValueCritic
from utils import make_d4rl_score, save_rollout_video

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages

def reinflow_finetune(env_name: str = "Hopper-v5", epochs: int = 100, rollout_steps: int = 1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = DiT1D(state_dim, action_dim)
    model.load_state_dict(torch.load(f"hseaf_{env_name.lower()}_offline.pt", weights_only=True))
    critic = ValueCritic(state_dim)
    critic_target = ValueCritic(state_dim)
    critic_target.load_state_dict(critic.state_dict())
    optimizer_policy = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    ds_id = f"mujoco/{env_name.lower().replace('-v5','')}/medium-v0"
    if "antmaze" in env_name.lower():
        ds_id = "D4RL/antmaze/large-play-v1"
    dataset = minari.load_dataset(ds_id)

    model.train()
    for epoch in tqdm(range(epochs), desc="ReinFlow Fine-tune"):
        obs, _ = env.reset()
        rollout_states, rollout_actions, rollout_rewards, rollout_dones, rollout_values = [], [], [], [], []
        for _ in range(rollout_steps):
            t = torch.rand(1)
            noise = torch.randn((1, action_dim))
            xt = noise
            out = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), xt, t)
            action = out["velocity"] + xt
            if out["meta_score"].item() < 1.05:
                action += out["escape"]
            rollout_states.append(obs)
            rollout_actions.append(action.squeeze(0).detach())
            obs, rew, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            rollout_rewards.append(rew)
            rollout_dones.append(terminated or truncated)
            v1, v2 = critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            rollout_values.append(min(v1, v2).item())
            if terminated or truncated:
                obs, _ = env.reset()

        states_t = torch.tensor(np.array(rollout_states), dtype=torch.float32)
        values = critic(states_t)[0].squeeze(-1).detach()
        advantages = compute_gae(torch.tensor(rollout_rewards), torch.tensor(rollout_values), torch.tensor(rollout_dones))
        returns = advantages + values

        v1, v2 = critic(states_t)
        v1_t, v2_t = critic_target.target_forward(states_t)
        v_loss = F.huber_loss(torch.clip(v1, values-1, values+1), returns, delta=1.0) + \
                 F.huber_loss(torch.clip(v2, values-1, values+1), returns, delta=1.0)
        optimizer_critic.zero_grad()
        v_loss.backward()
        optimizer_critic.step()
        critic.update_targets(tau=0.005)

        out = model(states_t, torch.randn_like(torch.tensor(rollout_actions)), torch.rand(len(states_t)))
        vel_loss = F.mse_loss(out["velocity"], torch.tensor(rollout_actions))
        simpo_weight = torch.sigmoid(advantages)
        w2_gap = (out["velocity"].mean(0) - torch.tensor(rollout_actions).mean(0)).pow(2).mean()
        policy_loss = (simpo_weight * vel_loss).mean() + 0.01 * w2_gap + 0.1 * out["completion"].mean()
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        if epoch % 10 == 0:
            score = make_d4rl_score(env_name, np.array(rollout_rewards))
            print(f"Epoch {epoch} | D4RL Score: {score:.1f}% | Meta: {out['meta_score'].mean().item():.3f}")

    torch.save({"model": model.state_dict(), "critic": critic.state_dict()}, f"hseaf_{env_name.lower()}_finetune.pt")
    save_rollout_video(lambda o: model(torch.tensor(o).unsqueeze(0), torch.randn(1, action_dim), torch.rand(1))["velocity"].squeeze(0).detach().numpy(), env_name)
    print("✅ ReinFlow fine-tuning complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    args = parser.parse_args()
    reinflow_finetune(args.env)

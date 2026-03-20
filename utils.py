import numpy as np
import imageio
import gymnasium as gym
import minari
from typing import List, Callable

def make_d4rl_score(env_name: str, returns: np.ndarray) -> float:
    returns = np.asarray(returns).flatten()
    try:
        ds_name = f"mujoco/{env_name.lower().replace('-v5','')}/medium-v0"
        if 'antmaze' in env_name.lower():
            ds_name = "D4RL/antmaze/large-play-v1"
        dataset = minari.load_dataset(ds_name)
        norm = minari.get_normalized_score(dataset, returns)
        return float(norm.mean() * 100)
    except:
        refs = {'halfcheetah': (-280, 12000), 'hopper': (-20, 3234), 'walker2d': (-20, 6000), 'ant': (-300, 6000), 'humanoid': (-200, 10000), 'antmaze': (0, 1)}
        key = next((k for k in refs if k in env_name.lower()), None)
        if key:
            rmin, rmax = refs[key]
            return float(((returns.mean() - rmin) / (rmax - rmin)) * 100)
        return float(returns.mean())

def save_rollout_video(policy: Callable, env_name: str = "Hopper-v5", filename: str = "rollout.mp4"):
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    obs, _ = env.reset()
    for _ in range(1000):
        action = policy(obs)
        frame = env.render()
        frames.append(frame)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved: {filename}")

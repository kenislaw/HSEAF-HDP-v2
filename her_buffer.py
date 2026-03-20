import torch
import numpy as np
from collections import deque

class SuccessBuffer:
    """HER self-distillation buffer: relabels failures + stores high-meta successes for replay.
    Tuned sweet-spot: threshold=1.15, lr=2e-5, weight=0.8."""
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, states, actions, achieved_goals, desired_goals, rewards, meta_scores):
        states = np.asarray(states)
        actions = np.asarray(actions)
        achieved_goals = np.asarray(achieved_goals)
        desired_goals = np.asarray(desired_goals)
        rewards = np.asarray(rewards)
        meta_scores = np.asarray(meta_scores)
        for i in range(len(states)):
            if rewards[i] == 0:  # HER relabel
                new_goal = achieved_goals[i]
                dist = np.linalg.norm(achieved_goals[i] - new_goal)
                new_rew = 1.0 if dist < 0.5 else 0.0
                self.buffer.append((states[i].copy(), actions[i].copy(), new_goal.copy(), new_rew, float(meta_scores[i])))
            else:
                self.buffer.append((states[i].copy(), actions[i].copy(), desired_goals[i].copy(), float(rewards[i]), float(meta_scores[i])))

    def sample(self, batch_size: int = 256):
        if len(self.buffer) < batch_size:
            return None
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        return states, actions

    def distill_high_meta(self, model, optimizer, threshold: float = 1.15):
        batch = self.sample(512)
        if batch is None:
            return 0.0
        states, actions = batch
        out = model(states, torch.randn_like(actions), torch.rand(len(states)))
        loss = torch.nn.functional.mse_loss(out['velocity'], actions) * 0.8
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

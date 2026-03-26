import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import gymnasium as gym
import sys
import copy
import time
from datetime import timedelta

# ==================== EMA ====================
class EMA:
    def __init__(self, model, decay=0.9997, warm_up_steps=1000):
        self.decay = decay
        self.warm_up_steps = warm_up_steps
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.step = 0

    def update(self, model):
        self.step += 1
        current_decay = min(self.decay, 0.99 + (self.decay - 0.99) * (self.step / self.warm_up_steps))
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(current_decay).add_(p.data, alpha=1 - current_decay)

    def eval(self):
        self.ema_model.eval()
        return self.ema_model


# ==================== CHECKPOINT ====================
class CheckpointManager:
    def __init__(self):
        self.model = None
        self.ema_model = None
        self.optimizer = None
        self.epoch = 0
        self.loss = 0.0
        self.writer = None
        self.args = None

    def save(self):
        if self.model is None:
            return
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/{self.args.env}_dit_latest.pt"
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict() if self.ema_model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'loss': self.loss,
        }, ckpt_path)
        print(f"✅ Checkpoint saved → {ckpt_path}")
        if self.writer:
            self.writer.close()
            print("✅ TensorBoard writer closed")


manager = CheckpointManager()


# ==================== DiT (RTG-conditioned BC) ====================
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

    def forward(self, x, t_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.mlp(x_norm)
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x.contiguous()


class DiT(nn.Module):
    def __init__(self, hidden_dim=256, depth=6, state_dim=11, action_dim=3, rtg_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.rtg_embed = nn.Linear(rtg_dim, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.head.bias.data.zero_()

    def forward(self, state, rtg):
        if rtg.dim() == 1:
            rtg = rtg.unsqueeze(-1)
        t = torch.zeros(state.shape[0], device=state.device)
        t_emb = self.time_embed(t.unsqueeze(-1).expand(-1, self.hidden_dim))
        
        x = self.state_embed(state) + self.rtg_embed(rtg)
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.norm(x.squeeze(1))
        return self.head(x)


# ==================== LOSSES ====================
def action_reg_loss(pred):
    return 0.001 * torch.mean(pred ** 2)

def variance_matching_loss(pred):
    return 0.25 * torch.mean((pred.std(dim=0) - 1.0) ** 2)

def mean_matching_loss(pred):
    return 0.05 * torch.mean(pred ** 2)

def reward_weighted_bc_loss(pred, target, bc_weight=8.0):
    return bc_weight * F.mse_loss(pred, target, reduction='mean')


# ==================== NORMALIZATION (warning-free) ====================
state_mean = None
state_std = None
action_mean = None
action_std = None
dataset_action_mean = None
dataset_action_std = None
rtg_min = 0.0   # will be set as float
rtg_max = 3.9   # will be set as float

def normalize_states(states):
    global state_mean, state_std
    if state_mean is None:
        mean_np = states.mean(axis=0)
        std_np = states.std(axis=0).clip(min=1e-4)
        state_mean = torch.from_numpy(mean_np).float()
        state_std = torch.from_numpy(std_np).float()
    if isinstance(states, torch.Tensor):
        return (states - state_mean.to(states.device)) / state_std.to(states.device)
    return (states - state_mean.numpy()) / state_std.numpy()

def normalize_actions(actions):
    global action_mean, action_std, dataset_action_mean, dataset_action_std
    if action_mean is None:
        mean_np = actions.mean(axis=0)
        std_np = actions.std(axis=0).clip(min=1e-4)
        action_mean = torch.from_numpy(mean_np).float()
        action_std = torch.from_numpy(std_np).float()
        dataset_action_mean = mean_np
        dataset_action_std = std_np
    if isinstance(actions, torch.Tensor):
        return (actions - action_mean.to(actions.device)) / action_std.to(actions.device)
    return (actions - action_mean.numpy()) / action_std.numpy()

def denormalize_actions(norm_actions):
    global action_mean, action_std
    if isinstance(norm_actions, torch.Tensor):
        return norm_actions * action_std.to(norm_actions.device) + action_mean.to(norm_actions.device)
    return norm_actions * action_std.numpy() + action_mean.numpy()

def normalize_rtg(rtg):
    """Warning-free RTG normalization"""
    global rtg_min, rtg_max
    if isinstance(rtg, torch.Tensor):
        # Create fresh scalars on correct device
        min_t = torch.tensor(rtg_min, device=rtg.device, dtype=rtg.dtype)
        max_t = torch.tensor(rtg_max, device=rtg.device, dtype=rtg.dtype)
        return (rtg - min_t) / (max_t - min_t + 1e-6)
    else:
        return (rtg - rtg_min) / (rtg_max - rtg_min + 1e-6)


# ==================== EVAL ====================
last_eval_return = "N/A"
last_eval_std = "N/A"
last_eval_height = 0.0
last_eval_vel = 0.0
last_eval_mean_action = None
last_eval_length = 0.0
last_eval_ctrl_cost = 0.0
last_eval_act_std = None
last_eval_leg_torque = 0.0

def evaluate(model, env, num_episodes=10):
    global last_eval_return, last_eval_std, last_eval_height, last_eval_vel, last_eval_mean_action, last_eval_length, last_eval_ctrl_cost, last_eval_act_std, last_eval_leg_torque
    returns = []
    heights = []
    vels = []
    action_means = []
    action_stds = []
    lengths = []
    ctrl_costs = []
    device = next(model.parameters()).device

    rtg_val = 1.0
    rtg_tensor = torch.tensor([[rtg_val]], dtype=torch.float32, device=device)

    for _ in range(num_episodes):
        state, _ = env.reset()
        total_r = 0.0
        max_h = 0.0
        total_vel = 0.0
        total_action = np.zeros(3)
        total_ctrl = 0.0
        steps = 0

        for _ in range(1000):
            state_norm = normalize_states(torch.from_numpy(state).float().unsqueeze(0).to(device))
            with torch.no_grad():
                action_norm = model(state_norm, rtg_tensor)
                action_norm += 0.08 * torch.randn_like(action_norm)

            action_denorm = denormalize_actions(action_norm.squeeze(0)).cpu().numpy()
            action_denorm = np.clip(action_denorm, -1.0, 1.0)

            state, r, terminated, truncated, _ = env.step(action_denorm)

            total_r += float(r) if not np.isnan(r) and not np.isinf(r) else 0.0
            max_h = max(max_h, float(state[0]))
            total_vel += float(state[5])
            total_action += action_denorm
            total_ctrl += 0.001 * np.sum(action_denorm ** 2)
            steps += 1

            if terminated or truncated:
                break

        returns.append(total_r)
        heights.append(max_h)
        vels.append(total_vel / max(steps, 1))
        action_means.append(total_action / max(steps, 1))
        action_stds.append(np.std(total_action / max(steps, 1), axis=0))
        lengths.append(steps)
        ctrl_costs.append(total_ctrl / max(steps, 1))

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    last_eval_return = f"{mean_ret:.1f}"
    last_eval_std = f"{std_ret:.1f}"
    last_eval_height = np.mean(heights)
    last_eval_vel = np.mean(vels)
    last_eval_mean_action = np.mean(action_means, axis=0)
    last_eval_length = np.mean(lengths)
    last_eval_ctrl_cost = np.mean(ctrl_costs)
    last_eval_act_std = np.mean(action_stds)
    last_eval_leg_torque = last_eval_mean_action[1]

    return mean_ret, std_ret, last_eval_height


# ==================== TRAINING ====================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    print("🧹 Cleared CUDA cache")

    cache_path = "data/hopper_medium.npz"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(cache_path):
        print("Loading cached dataset...")
        data = np.load(cache_path)
        states = data['states']
        actions_np = data['actions']
        rtgs_raw = data['rtgs']
        if rtgs_raw.ndim > 1:
            rtgs_raw = rtgs_raw.flatten()
        rtgs = rtgs_raw.reshape(-1, 1)

        global state_mean, state_std, action_mean, action_std, dataset_action_mean, dataset_action_std, rtg_min, rtg_max
        state_mean = torch.from_numpy(states.mean(axis=0)).float()
        state_std = torch.from_numpy(states.std(axis=0)).float().clamp(min=1e-4)
        action_mean = torch.from_numpy(actions_np.mean(axis=0)).float()
        action_std = torch.from_numpy(actions_np.std(axis=0)).float().clamp(min=1e-4)
        dataset_action_mean = actions_np.mean(axis=0)
        dataset_action_std = actions_np.std(axis=0)
        rtg_min = float(rtgs_raw.min())
        rtg_max = float(rtgs_raw.max())

        print(f"State norm: mean={state_mean.numpy()[:5]}... std={state_std.numpy()[:5]}...")
        print(f"Action normalization: mean={action_mean.numpy()}, std={action_std.numpy()}")
        print(f"Dataset mean action[0] (thigh): {action_mean[0].item():.4f}")
        print(f"RTG range: {rtg_min:.1f} to {rtg_max:.1f}")
    else:
        raise FileNotFoundError("Cached dataset not found.")

    ds = TensorDataset(torch.from_numpy(states), torch.from_numpy(actions_np), torch.from_numpy(rtgs))

    loader = DataLoader(ds, batch_size=16384, shuffle=True,
                        num_workers=6, pin_memory=True, drop_last=True,
                        prefetch_factor=4, persistent_workers=True)

    model = DiT(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    ema = EMA(model, decay=0.9997)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: hidden={args.hidden_dim} depth={args.depth} | {param_count:,} params (RTG-conditioned DiT BC)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device="cuda")

    eval_env = gym.make(args.env)

    writer = SummaryWriter(log_dir=f"runs/{args.env}_dit_bc")
    manager.model = model
    manager.ema_model = ema.ema_model
    manager.optimizer = optimizer
    manager.writer = writer
    manager.args = args

    print("🚀 Training started (Strong RTG-conditioned BC – warning-free)")

    start_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            step_count = 0
            epoch_start = time.time()

            for state_b, action_b, rtg_b in loader:
                state_b = state_b.to(device, non_blocking=True)
                action_b = action_b.to(device, non_blocking=True)
                rtg_b = rtg_b.to(device, non_blocking=True)

                state_norm = normalize_states(state_b)
                action_norm = normalize_actions(action_b)
                rtg_norm = normalize_rtg(rtg_b)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(state_norm, rtg_norm)
                    
                    bc_l = reward_weighted_bc_loss(pred, action_norm, bc_weight=8.0)
                    reg_l = action_reg_loss(pred)
                    mean_l = mean_matching_loss(pred)
                    var_l = variance_matching_loss(pred)
                    loss = bc_l + reg_l + mean_l + var_l

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                epoch_loss += loss.item()
                step_count += 1

            scheduler.step()

            if epoch % 20 == 0:
                elapsed = time.time() - start_time
                epoch_time = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Loss {epoch_loss/step_count:.4f} | LR {current_lr:.2e} | it/s {step_count/epoch_time:.1f} | Time {timedelta(seconds=int(elapsed))}")

                ema_model = ema.eval()
                ret_mean, ret_std, avg_h = evaluate(ema_model, eval_env)
                leg_sign = "POSITIVE" if last_eval_leg_torque > 0 else "NEGATIVE"
                print(f"   → Eval Return {ret_mean:.1f}±{ret_std:.1f} | AvgTorsoHeight {avg_h:.3f} | AvgVel {last_eval_vel:.3f} | "
                      f"AvgEpisodeLength {last_eval_length:.0f} | EstCtrlCost {last_eval_ctrl_cost:.3f} | "
                      f"MeanAct {last_eval_mean_action} | ActStd {last_eval_act_std:.3f} | LegTorque {last_eval_leg_torque:.3f} ({leg_sign})")
                print(f"   → Debug: Dataset mean {dataset_action_mean} std {dataset_action_std} | Model mean {last_eval_mean_action}")

            manager.epoch = epoch
            manager.loss = epoch_loss / step_count

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Graceful shutdown...")
        manager.save()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        manager.save()
        raise

    manager.save()
    writer.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v5")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--meta-decouple-factor", type=float, default=0.0)
    parser.add_argument("--bc-weight", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()

    train(args)

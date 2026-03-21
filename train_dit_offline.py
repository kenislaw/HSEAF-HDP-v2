import os
import time
import torch
import torch.nn.functional as F
import minari
import gymnasium as gym
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import copy

torch.serialization.add_safe_globals([
    np._core.multiarray.scalar,
    np.dtype,
    np.dtypes.Float64DType
])

from models import DiT1D
from utils import make_d4rl_score

class MinariTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
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
            raise RuntimeError(f"Length mismatch: states={len(self.states)}, actions={len(self.actions)}")
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(device)
        print(f"Pre-loaded dataset to GPU ({device}) | {len(self.states):,} transitions")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model)
        self.decay = decay

    def update(self, model):
        for p, s in zip(model.parameters(), self.shadow.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

class MetaController:
    def __init__(self, hyper_params=None):
        self.meta_weight = 0.05
        self.meta_weight_floor = 0.05
        self.beta_alpha = 5.0
        self.beta_beta = 1.0
        self.stagnation_counter = 0
        self.best_return = -float("inf")
        self.last_returns = []
        self.hyper = hyper_params or {}

    def update(self, avg_return, avg_loss):
        self.last_returns.append(avg_return)
        if len(self.last_returns) > 10:
            self.last_returns.pop(0)
        recent_avg = np.mean(self.last_returns[-5:]) if self.last_returns else avg_return
        if recent_avg > self.best_return:
            self.best_return = recent_avg
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if self.stagnation_counter > self.hyper.get("stagnation_threshold", 8):
            self.meta_weight = min(0.3, self.meta_weight * 1.20)
            self.beta_alpha = max(2.0, self.beta_alpha * 0.90)
            if self.stagnation_counter > 12:
                print("  → Stagnation detected: triggering warm LR restart")
                self.stagnation_counter = 0
                return True
        else:
            self.meta_weight = max(self.meta_weight_floor, self.meta_weight * 0.96)
            self.beta_alpha = min(8.0, self.beta_alpha * 1.04)
        return False

class HyperMetaController:
    def __init__(self):
        self.stagnation_threshold = 8
        self.clamp_max = 8.0
        self.clamp_min = 0.01          # FINAL: force activation
        self.grace_epochs = 12
        self.ema_decay = 0.999

    def update(self, avg_return, loss_std, return_std):
        meta_reward = (avg_return / (return_std + 1e-6)) - (loss_std * 8)
        if meta_reward > 5:
            self.clamp_min = max(0.005, self.clamp_min * 0.92)  # dynamic drop
            self.stagnation_threshold = max(4, min(12, int(self.stagnation_threshold * 0.97)))
            self.grace_epochs = max(8, min(25, int(self.grace_epochs * 0.98)))
        return {
            "stagnation_threshold": self.stagnation_threshold,
            "clamp_max": self.clamp_max,
            "grace_epochs": self.grace_epochs,
            "ema_decay": self.ema_decay
        }

class OuterController:
    def __init__(self):
        self.last_losses = []
        self.last_returns = []
        self.last_meta_weights = []
        self.return_ema = 0.0
        self.ema_alpha = 0.85

    def update(self, avg_loss, avg_return, current_meta_weight, return_std):
        self.last_losses.append(avg_loss)
        self.last_returns.append(avg_return)
        self.last_meta_weights.append(current_meta_weight)
        if len(self.last_losses) > 40:
            self.last_losses.pop(0)
            self.last_returns.pop(0)
            self.last_meta_weights.pop(0)
        loss_std = np.std(self.last_losses) if len(self.last_losses) > 15 else 0

        self.return_ema = self.ema_alpha * self.return_ema + (1 - self.ema_alpha) * avg_return

        spike = avg_loss > 0.45
        high_vol = return_std > 80
        if spike or high_vol:
            print("  → OuterController: SPIKE or HIGH VOL – forcing aggressive warm restart")
            return True, loss_std, return_std, True

        if avg_loss < 0.35 and avg_return > self.return_ema + 25 and return_std < 40:
            print("  → OuterController: Strong stable improvement + low vol – boosting meta_weight + floor")
            return False, loss_std, return_std, True
        return False, loss_std, return_std, False

def train_dit_offline(
    env_name: str = "Hopper-v5",
    epochs: int = 100,
    micro_batch_size: int = 131072,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    eval_freq: int = 20,
    num_workers: int = 4,
):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    ds_id = f"mujoco/{env_name.lower().replace('-v5', '')}/medium-v0"
    if "antmaze" in env_name.lower():
        ds_id = "D4RL/antmaze/large-play-v1"

    dataset = minari.load_dataset(ds_id, download=True)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    effective_batch = micro_batch_size * grad_accum_steps
    lr_scaled = lr * (effective_batch ** 0.5 / 256.0)
    print(f"Effective batch size: {effective_batch} | Scaled LR: {lr_scaled:.2e}")

    model = DiT1D(state_dim, action_dim).to(device)
    ema = EMA(model)
    outer_ctrl = OuterController()
    hyper_ctrl = HyperMetaController()
    meta_ctrl = MetaController(hyper_params=hyper_ctrl.update(0, 0, 0))

    print(f"Model params device: {next(model.parameters()).device}")
    print(f"Total model params: {sum(p.numel() for p in model.parameters()):,}")

    print("Running in eager mode (Triton unavailable on Windows)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_scaled, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # FINAL scheduler: long Cosine + plateau on meta_loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=5e-6)
    meta_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=5e-6)

    ckpt_dir = f"checkpoints/{env_name.lower()}"
    os.makedirs(ckpt_dir, exist_ok=True)

    final_path = f"hseaf_{env_name.lower()}_offline.pt"
    best_path = f"hseaf_{env_name.lower()}_offline_best.pt"

    best_score = -float("inf")
    start_epoch = 0
    resume_path = None

    latest_epoch = -1
    for f in os.listdir(ckpt_dir):
        if f.startswith("interrupted_epoch_"):
            try:
                ep_num = int(f.split("_")[2])
                if ep_num > latest_epoch:
                    latest_epoch = ep_num
                    resume_path = os.path.join(ckpt_dir, f)
            except:
                pass

    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            ema.shadow.load_state_dict(ckpt.get("ema", ckpt["model"]))
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            model = model.to(device)
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed INTERRUPTED checkpoint → {resume_path} (epoch {start_epoch})")
        except Exception as e:
            print(f"Interrupted resume failed ({e}) → starting fresh")

    traj_ds = MinariTrajectoryDataset(dataset, device)
    loader = DataLoader(
        traj_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    def quick_eval(model_to_eval, env, n_ep=20):
        model_to_eval.eval()
        returns = []
        for _ in range(n_ep):
            obs, _ = env.reset()
            ep_ret = 0.0
            done = False
            step = 0
            while not done and step < 1000:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                    t = torch.zeros(1, device=device)
                    xt = torch.randn(action_dim, device=device).unsqueeze(0) * 1.0
                    for _ in range(6):
                        out = model_to_eval(obs_t, xt, t)
                        xt = xt - out["velocity"] * 0.25
                    act = out["velocity"][0].cpu().numpy()
                obs, rew, terminated, truncated, _ = env.step(act)
                ep_ret += rew
                done = terminated or truncated
                step += 1
            returns.append(ep_ret)
        model_to_eval.train()
        ret_mean = np.mean(returns)
        ret_std = np.std(returns)
        return ret_mean, ret_std

    model.train()
    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            vel_loss_sum = 0.0
            meta_loss_sum = 0.0
            n_samples = 0
            accum_counter = 0

            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
            for i, (states, actions) in enumerate(pbar):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if epoch < start_epoch + hyper_ctrl.grace_epochs:
                        t = torch.rand(len(states), device=device)
                    else:
                        t = torch.distributions.Beta(meta_ctrl.beta_alpha, meta_ctrl.beta_beta).sample((len(states),)).to(device)

                    noise = torch.randn_like(actions)
                    xt = (1 - t[:, None]) * noise + t[:, None] * actions
                    out = model(states, xt, t)

                    target_vel = actions - noise
                    vel_loss = F.mse_loss(out["velocity"], target_vel)
                    meta_loss_raw = (out["meta_score"] - 1.0).pow(2).mean()
                    meta_loss = torch.clamp(meta_loss_raw, max=hyper_ctrl.clamp_max, min=hyper_ctrl.clamp_min)
                    loss = vel_loss + meta_ctrl.meta_weight * meta_loss

                scaler.scale(loss / grad_accum_steps).backward()
                accum_counter += 1

                if accum_counter == grad_accum_steps or (i + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    scheduler.step()
                    meta_plateau.step(meta_loss_raw.item())  # force meta contribution

                    ema.update(model)
                    accum_counter = 0

                epoch_loss += loss.item() * states.size(0) * grad_accum_steps
                vel_loss_sum += vel_loss.item() * states.size(0)
                meta_loss_sum += meta_loss_raw.item() * states.size(0)
                n_samples += states.size(0)

            avg_loss = epoch_loss / n_samples
            avg_vel = vel_loss_sum / n_samples
            avg_meta = meta_loss_sum / n_samples
            peak_vram = torch.cuda.max_memory_allocated() / 1e9

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} Loss: {avg_loss:.6f} (vel: {avg_vel:.6f} | meta: {avg_meta:.6f}) | Peak VRAM: {peak_vram:.1f} GB | LR: {current_lr:.2e}")
            print(f"  MetaWeight: {meta_ctrl.meta_weight:.3f} | Betaα: {meta_ctrl.beta_alpha:.1f} | HyperClamp: {hyper_ctrl.clamp_max:.1f} | MetaFloor: {meta_ctrl.meta_weight_floor:.3f}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if (epoch + 1) % eval_freq == 0:
                eval_start = time.perf_counter()
                avg_ret, ret_std = quick_eval(ema.shadow, env)
                eval_time = time.perf_counter() - eval_start
                ckpt_path = f"{ckpt_dir}/epoch_{(epoch+1):04d}_ret{avg_ret:.1f}.pt"
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "avg_return": avg_ret,
                }
                torch.save(checkpoint, ckpt_path)
                print(f"Checkpoint → {ckpt_path} | AvgReturn: {avg_ret:.2f} ±{ret_std:.1f} | Eval time: {eval_time:.1f}s")

                score = avg_ret - 0.3 * ret_std
                if score > best_score:
                    best_score = score
                    torch.save(checkpoint, best_path)
                    print("  → New BEST (volatility-penalized) model saved")

                spike, loss_std, return_std, boost_meta = outer_ctrl.update(avg_loss, avg_ret, meta_ctrl.meta_weight, ret_std)
                if spike or ret_std > 80:
                    print("  → OuterController: SPIKE or HIGH VOL – resetting scheduler")
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=5e-6)
                if boost_meta:
                    meta_ctrl.meta_weight = min(0.3, meta_ctrl.meta_weight * 1.22)
                    meta_ctrl.meta_weight_floor = max(0.05, meta_ctrl.meta_weight_floor * 1.12)

                hyper_params = hyper_ctrl.update(avg_ret, loss_std, return_std)
                meta_ctrl.hyper = hyper_params

                if meta_ctrl.update(avg_ret, avg_loss):
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=5e-6)

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt detected. Saving interrupted checkpoint...")
        interrupted_path = f"{ckpt_dir}/interrupted_epoch_{epoch+1}_ret{avg_ret if 'avg_ret' in locals() else 'unknown'}.pt"
        checkpoint = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss if 'avg_loss' in locals() else 0.0,
            "avg_return": avg_ret if 'avg_ret' in locals() else 0.0,
        }
        torch.save(checkpoint, interrupted_path)
        print(f"Interrupted state saved → {interrupted_path}")
        raise

    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch + 1,
        "loss": avg_loss,
        "avg_return": avg_ret if "avg_ret" in locals() else 0.0,
    }, final_path)
    print(f"Training complete → {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline DiT pre-training - HSEAF-HDP v2")
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=131072)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train_dit_offline(
        env_name=args.env,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        eval_freq=args.eval_freq,
        num_workers=args.num_workers,
    )

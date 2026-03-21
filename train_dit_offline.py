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
import platform

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
    print(f"Model params device: {next(model.parameters()).device}")
    print(f"Total model params: {sum(p.numel() for p in model.parameters()):,}")

    print("Running in eager mode (Triton unavailable on Windows)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_scaled)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # === Scheduler (warmup → cosine) ===
    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr_scaled * 0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    ckpt_dir = f"checkpoints/{env_name.lower()}"
    os.makedirs(ckpt_dir, exist_ok=True)

    final_path = f"hseaf_{env_name.lower()}_offline.pt"
    best_path = f"hseaf_{env_name.lower()}_offline_best.pt"

    best_return = -float("inf")
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
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_scaled)
            start_epoch = ckpt.get("epoch", 0)
            best_return = ckpt.get("avg_return", -float("inf"))
            print(f"Resumed INTERRUPTED checkpoint → {resume_path} (epoch {start_epoch})")
        except Exception as e:
            print(f"Interrupted resume failed: {e}")
    elif os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_scaled)
            best_return = ckpt.get("avg_return", -float("inf"))
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed BEST checkpoint (return: {best_return:.2f})")
        except Exception as e:
            print(f"Best resume failed: {e} → starting fresh")

    traj_ds = MinariTrajectoryDataset(dataset, device)
    loader = DataLoader(
        traj_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    def quick_eval(model, env, n_ep=5, max_steps=1000):
        model.eval()
        returns = []
        for _ in range(n_ep):
            obs, _ = env.reset()
            ep_ret = 0.0
            step = 0
            done = False
            while not done and step < max_steps:
                with torch.no_grad():
                    t = torch.zeros(1, device=device)
                    noise = torch.randn(action_dim, device=device)
                    xt = noise
                    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                    out = model(obs_t, xt.unsqueeze(0), t)
                    act = out["velocity"][0].cpu().numpy()
                obs, rew, terminated, truncated, _ = env.step(act)
                done = terminated or truncated
                ep_ret += rew
                step += 1
            returns.append(ep_ret)
        model.train()
        return sum(returns) / len(returns) if returns else 0.0

    model.train()
    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            n_samples = 0
            forward_time = 0.0
            backward_time = 0.0

            optimizer.zero_grad(set_to_none=True)

            for i, (states, actions) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
                fwd_start = time.perf_counter()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    t = torch.rand(len(states), device=device)
                    noise = torch.randn_like(actions)
                    xt = (1 - t[:, None]) * noise + t[:, None] * actions
                    out = model(states, xt, t)

                    target_vel = actions - noise
                    vel_loss = F.mse_loss(out["velocity"], target_vel)
                    meta_loss = (out["meta_score"] - 1.0).pow(2).mean()
                    loss = vel_loss + 0.1 * meta_loss
                forward_time += time.perf_counter() - fwd_start

                bwd_start = time.perf_counter()
                scaler.scale(loss / grad_accum_steps).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                backward_time += time.perf_counter() - bwd_start

                epoch_loss += loss.item() * states.size(0) * grad_accum_steps
                n_samples += states.size(0)

                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / n_samples if n_samples > 0 else 0.0
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            epoch_time = time.perf_counter() - epoch_start

            print(f"Epoch {epoch+1} Loss: {avg_loss:.6f} | Peak VRAM: {peak_vram:.1f} GB")
            print(f"  Time: {epoch_time:.1f}s total | Forward: {forward_time:.1f}s | Backward: {backward_time:.1f}s")

            # === CORRECT SCHEDULER PLACEMENT: AFTER full accumulation & optimizer.step() ===
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if (epoch + 1) % eval_freq == 0:
                eval_start = time.perf_counter()
                avg_ret = quick_eval(model, env)
                eval_time = time.perf_counter() - eval_start
                ckpt_path = f"{ckpt_dir}/epoch_{(epoch+1):04d}_ret{avg_ret:.1f}.pt"
                checkpoint = {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "avg_return": avg_ret,
                }
                torch.save(checkpoint, ckpt_path)
                print(f"Checkpoint → {ckpt_path} | AvgReturn: {avg_ret:.2f} | Eval time: {eval_time:.1f}s")

                if avg_ret > best_return:
                    best_return = avg_ret
                    torch.save(checkpoint, best_path)
                    print("  → New BEST model saved")

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt detected. Saving interrupted checkpoint...")
        interrupted_path = f"{ckpt_dir}/interrupted_epoch_{epoch+1}_ret{avg_ret if 'avg_ret' in locals() else 'unknown'}.pt"
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss if 'avg_loss' in locals() else 0.0,
            "avg_return": avg_ret if 'avg_ret' in locals() else 0.0,
        }
        torch.save(checkpoint, interrupted_path)
        print(f"Interrupted state saved → {interrupted_path}")
        raise

    torch.save({
        "model": model.state_dict(),
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

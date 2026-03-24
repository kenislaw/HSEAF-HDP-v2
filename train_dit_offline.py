import os
import copy
import torch
import torch.nn.functional as F
import minari
import gymnasium as gym
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import warnings

# ====================== GLOBAL SETTINGS ======================
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ====================== CONFIG ======================
@dataclass
class Config:
    env_name: str = "Hopper-v5"
    epochs: int = 2000
    micro_batch: int = 16384
    grad_accum: int = 8
    lr: float = 3e-4
    eval_freq: int = 5
    target_rtg: float = 950.0
    inference_steps: int = 30
    device: torch.device = torch.device("cuda")
    embed_dim: int = 256
    depth: int = 6
    meta_start: int = 30


def cosine_beta_schedule(timesteps: int = 1000, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# ====================== MODEL ======================
class DiT1D(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, embed_dim: int = 256, depth: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_embed = torch.nn.Linear(state_dim, embed_dim)
        self.action_embed = torch.nn.Linear(action_dim, embed_dim)
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim), torch.nn.SiLU(), torch.nn.Linear(embed_dim, embed_dim)
        )
        self.rtg_proj = torch.nn.Linear(1, embed_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=depth)
        self.eps_head = torch.nn.Linear(embed_dim, action_dim)
        self.meta_head = torch.nn.Linear(embed_dim, 1)

    def forward(self, state, x_t, t, rtg=None):
        s_emb = self.state_embed(state)
        if rtg is not None:
            s_emb = s_emb + self.rtg_proj(rtg.unsqueeze(-1))
        x_emb = self.action_embed(x_t)
        t_emb = self.time_embed(t.view(-1, 1))
        x = torch.stack([s_emb, x_emb, t_emb], dim=1)
        x = self.transformer(x)
        return {
            "eps": self.eps_head(x[:, 1, :]),
            "meta": self.meta_head(x.mean(dim=1)).squeeze(-1)
        }


# ====================== DATASET ======================
class MinariTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.states = []
        self.actions = []
        self.raw_returns = []
        for ep in dataset:
            min_len = min(len(ep.observations), len(ep.actions))
            if min_len == 0:
                continue
            self.states.extend(ep.observations[:min_len])
            self.actions.extend(ep.actions[:min_len])
            cum_ret = np.cumsum(getattr(ep, 'rewards', np.zeros(min_len))[::-1])[::-1]
            self.raw_returns.extend(cum_ret)

        self.states_np = np.array(self.states, dtype=np.float32)
        self.state_mean = self.states_np.mean(0)
        self.state_std = self.states_np.std(0) + 1e-6
        self.states = torch.tensor(
            (self.states_np - self.state_mean) / self.state_std, dtype=torch.float32
        ).to(device)

        self.actions_np = np.array(self.actions, dtype=np.float32)
        self.action_mean = self.actions_np.mean(0)
        self.action_std = self.actions_np.std(0) + 1e-6
        self.actions = torch.tensor(
            (self.actions_np - self.action_mean) / self.action_std, dtype=torch.float32
        ).to(device)

        returns_array = np.clip(np.array(self.raw_returns), 0, 3500)
        self.raw_returns = torch.tensor(returns_array, dtype=torch.float32).to(device)
        self.returns = self.raw_returns / 1000.0

        print(f"Pre-loaded dataset to GPU | {len(self.states):,} transitions "
              f"(returns clipped & scaled /1000)")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.returns[idx], self.raw_returns[idx]


# ====================== EMA ======================
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model)
        self.decay = decay

    def update(self, model):
        for p, s in zip(model.parameters(), self.shadow.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ====================== EVALUATOR ======================
class QuickEvaluator:
    def __init__(self, traj_ds, device, target_rtg=950.0):
        self.traj_ds = traj_ds
        self.device = device
        self.target_rtg = target_rtg / 1000.0
        self.betas = cosine_beta_schedule()
        self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0).to(device)

    def evaluate(self, model, env, n_ep=3, steps=30):
        model.eval()
        returns = []
        all_heights = []
        all_actions = []
        stride = max(1, 1000 // steps)
        rtg_t = torch.full((1,), self.target_rtg, device=self.device)

        print(f"\n{'='*70}")
        print(f"→ Eval {n_ep}×2 episodes (DDIM steps={steps})...")

        for ep_idx in range(n_ep):
            for s_idx in range(2):
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_heights = []
                ep_actions = []

                for _ in range(1000):
                    x_t = torch.randn((1, env.action_space.shape[0]), device=self.device)

                    for d in range(steps):
                        t_idx = 999 - d * stride
                        t = torch.full((1,), t_idx / 1000.0, device=self.device)
                        progress = d / steps
                        guidance = 0.5 + 4.0 * (1 - progress ** 0.6)

                        state_t = torch.from_numpy(
                            (obs - self.traj_ds.state_mean) / self.traj_ds.state_std
                        ).float().unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            out_c = model(state_t, x_t, t, rtg=rtg_t)
                            out_u = model(state_t, x_t, t, rtg=None)
                            eps = out_u["eps"] + guidance * (out_c["eps"] - out_u["eps"])

                        alpha_t = self.alphas_cumprod[t_idx]
                        alpha_p = self.alphas_cumprod[max(0, t_idx - stride)]
                        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
                        x_t = torch.sqrt(alpha_p) * pred_x0 + torch.sqrt(1 - alpha_p) * eps
                        x_t = x_t.clamp(-2.5, 2.5)

                    act_norm = x_t[0].cpu().numpy()
                    act = (act_norm * self.traj_ds.action_std + self.traj_ds.action_mean).clip(
                        env.action_space.low * 0.95, env.action_space.high * 0.95)

                    obs, r, term, trunc, _ = env.step(act)
                    ep_ret += r
                    ep_heights.append(obs[0])
                    ep_actions.append(np.abs(act).mean())

                    if term or trunc or obs[0] < 0.7:
                        break

                returns.append(ep_ret)
                all_heights.extend(ep_heights)
                all_actions.extend(ep_actions)

                print(f"  [{ep_idx}][{s_idx}] ret={ep_ret:7.1f} len={len(ep_heights):3d} "
                      f"max_h={max(ep_heights):.2f} act_norm={np.mean(ep_actions):.3f}")

        avg_ret = np.mean(returns)
        ret_std = np.std(returns)
        avg_h = np.mean(all_heights)
        h_min, h_max = np.min(all_heights), np.max(all_heights)
        avg_act = np.mean(all_actions)
        early_death_pct = 100 * sum(h < 0.7 for h in all_heights) / len(all_heights)

        bins = [0.0, 0.7, 1.0, 1.3, 1.6, 2.0]
        hist, _ = np.histogram(all_heights, bins=bins)
        hist_str = " | ".join([f"{bins[i]:.1f}-{bins[i+1]:.1f}:{hist[i]/len(all_heights)*100:4.0f}%" for i in range(len(hist))])

        print(f"{'-'*70}")
        print(f"DEBUG EVAL → avg_ret={avg_ret:6.1f}±{ret_std:.1f} | avg_height={avg_h:.2f} "
              f"(min={h_min:.2f}, max={h_max:.2f}) | action_norm={avg_act:.3f}")
        print(f"             early_death={early_death_pct:.1f}% | height_dist: {hist_str}")
        print(f"{'='*70}\n")

        model.train()
        return avg_ret


# ====================== MAIN TRAINER ======================
def train_dit_offline(cfg: Config):
    effective_batch = cfg.micro_batch * cfg.grad_accum
    lr_scaled = cfg.lr * (effective_batch ** 0.5 / 256)

    print(f"Using device: {cfg.device} | {torch.cuda.get_device_name(0)}")
    print(f"Effective batch size: {effective_batch} | Scaled LR: {lr_scaled:.2e}")

    ds_id = f"mujoco/{cfg.env_name.lower().replace('-v5', '')}/medium-v0"
    dataset = minari.load_dataset(ds_id, download=True)
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = DiT1D(state_dim, action_dim, cfg.embed_dim, cfg.depth).to(cfg.device)

    # Force eager mode - torch.compile is broken on RTX 5070 (Blackwell + Triton missing)
    print("torch.compile disabled (Blackwell GPU) → using eager mode for stability")

    ema = EMA(model)
    traj_ds = MinariTrajectoryDataset(dataset, cfg.device)
    loader = DataLoader(traj_ds, batch_size=cfg.micro_batch, shuffle=True, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_scaled, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, eta_min=1e-4)

    log_dir = f"runs/{cfg.env_name.lower()}_dit_v32"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging enabled → {log_dir}")
    print("   View with: tensorboard --logdir runs\n")

    evaluator = QuickEvaluator(traj_ds, cfg.device, cfg.target_rtg)

    meta_weight = 0.08
    print("Starting training...\n")

    try:
        for epoch in range(cfg.epochs):
            epoch_loss = eps_loss_sum = meta_loss_sum = n_samples = 0.0
            optimizer.zero_grad()

            pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}")
            for states, actions, _, raw_returns in pbar:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    t = torch.rand(len(states), device=cfg.device)
                    noise = torch.randn_like(actions)
                    sqrt_alpha = torch.sqrt(1 - t[:, None])
                    sqrt_one_minus_alpha = torch.sqrt(t[:, None])
                    x_t = sqrt_alpha * actions + sqrt_one_minus_alpha * noise

                    rtg = torch.full((len(states),), cfg.target_rtg / 1000.0, device=cfg.device) \
                          if torch.rand(1).item() < 0.6 else None

                    out = model(states, x_t, t, rtg=rtg)

                    eps_loss = F.mse_loss(out["eps"], noise)
                    meta_loss = F.mse_loss(out["meta"], (raw_returns / 1000.0 - 0.5) * 2.0)

                    active_meta = meta_weight if epoch >= cfg.meta_start else 0.0
                    loss = eps_loss + active_meta * meta_loss

                scaler.scale(loss).backward()

                if (n_samples // cfg.micro_batch + 1) % cfg.grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    ema.update(model)

                epoch_loss += loss.item()
                eps_loss_sum += eps_loss.item()
                meta_loss_sum += meta_loss.item()
                n_samples += len(states)

            avg_loss = epoch_loss / len(loader)
            avg_eps = eps_loss_sum / len(loader)
            avg_meta = meta_loss_sum / len(loader)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1:3d} | Loss {avg_loss:.5f} (eps: {avg_eps:.5f} | meta: {avg_meta:.4f}) "
                  f"| LR {current_lr:.2e} | MetaW {meta_weight:.3f} | GradNorm {grad_norm:.3f}")
            print(f"  DEBUG TRAIN → eps_norm={out['eps'].abs().mean().item():.3f}±{out['eps'].abs().std().item():.3f} "
                  f"| meta_range=[{out['meta'].min().item():.2f}, {out['meta'].max().item():.2f}]")

            if writer:
                writer.add_scalar("Loss/total", avg_loss, epoch)
                writer.add_scalar("Loss/eps", avg_eps, epoch)
                writer.add_scalar("Loss/meta", avg_meta, epoch)
                writer.add_scalar("Hyper/lr", current_lr, epoch)
                writer.add_scalar("Hyper/meta_weight", meta_weight, epoch)

            if (epoch + 1) % cfg.eval_freq == 0 or epoch == 0:
                avg_ret = evaluator.evaluate(ema.shadow, env, n_ep=3, steps=cfg.inference_steps)
                if writer:
                    writer.add_scalar("Eval/return_mean", avg_ret, epoch)

                if epoch >= cfg.meta_start:
                    if avg_ret > 100:
                        meta_weight = min(0.25, meta_weight * 1.03)
                    else:
                        meta_weight = max(0.08, meta_weight * 0.97)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt → saving interrupted checkpoint...")
        os.makedirs(f"checkpoints/{cfg.env_name.lower()}", exist_ok=True)
        torch.save(ema.shadow.state_dict(), f"checkpoints/{cfg.env_name.lower()}/interrupted.pt")
    finally:
        if writer:
            writer.close()

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--force-new", action="store_true")
    parser.add_argument("--inference-denoising-steps", type=int, default=30)
    parser.add_argument("--target-rtg", type=float, default=950.0)
    args = parser.parse_args()

    cfg = Config(
        env_name=args.env,
        epochs=args.epochs,
        inference_steps=args.inference_denoising_steps,
        target_rtg=args.target_rtg,
    )

    os.makedirs(f"checkpoints/{cfg.env_name.lower()}", exist_ok=True)
    train_dit_offline(cfg)

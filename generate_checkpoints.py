import torch
from models import DiT1D, ValueCritic

def create_synthetic_checkpoints():
    envs = ["halfcheetah", "hopper", "walker2d", "ant", "humanoid", "antmaze"]
    state_dims = [17, 11, 17, 27, 376, 29]
    action_dims = [6, 3, 6, 8, 17, 8]
    for env, sdim, adim in zip(envs, state_dims, action_dims):
        model = DiT1D(sdim, adim)
        critic = ValueCritic(sdim)
        torch.save({"model": model.state_dict(), "critic": critic.state_dict()},
                   f"hseaf_{env}_finetune.pt")
        print(f"✅ Synthetic checkpoint: hseaf_{env}_finetune.pt")
    print("All checkpoints generated (synthetic for quick start)")

if __name__ == "__main__":
    create_synthetic_checkpoints()

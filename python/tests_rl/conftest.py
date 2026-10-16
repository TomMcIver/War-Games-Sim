import numpy as np
import pytest
import torch

from warsim_rl.config import ModelConfig
from warsim_rl.mock_env import MockVecEnv
from warsim_rl.models import ActorCritic

torch.set_num_threads(1)


@pytest.fixture
def small_model_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=16, n_heads=2, n_layers=2, cnn_channels=(4, 8), cnn_out=16,
        hidden=32, critic_d=16, critic_hidden=32,
    )


@pytest.fixture
def small_model(small_model_cfg) -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(small_model_cfg)


def agent_batch(n_envs: int = 3, team_size: int = 2, seed: int = 0) -> dict:
    """Torch per-agent batch for every unit of a fresh mock env."""
    env = MockVecEnv(n_envs=n_envs, team_size=team_size, seed=seed)
    obs = env.reset()
    E, U = n_envs, env.n_units
    e_idx = np.repeat(np.arange(E), U)
    u_idx = np.tile(np.arange(U), E)
    return {
        "tokens": torch.as_tensor(obs["tokens"][e_idx, u_idx]),
        "token_mask": torch.as_tensor(obs["token_mask"][e_idx, u_idx]),
        "local_map": torch.as_tensor(obs["local_map"][e_idx, u_idx]),
        "alive": torch.as_tensor(obs["alive"][e_idx, u_idx]),
        "state": torch.as_tensor(obs["state"][e_idx]),
        "agent_idx": torch.as_tensor(u_idx),
    }

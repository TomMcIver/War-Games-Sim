import json

import pytest
import torch

from conftest import agent_batch
from warsim_rl.checkpoint import load_model, save_model
from warsim_rl.config import apply_overrides, config_from_dict, load_config
from warsim_rl.policies import IdlePolicy, NetPolicy, RandomPolicy


def test_checkpoint_round_trip(tmp_path, small_model):
    path = tmp_path / "m.pt"
    save_model(path, small_model, {"update": 3})
    loaded = load_model(path)
    assert loaded.cfg == small_model.cfg
    batch = agent_batch(2, 2)
    B = batch["tokens"].shape[0]
    h, r = small_model.initial_state(B), torch.ones(B)
    a = small_model.step(batch, h, r)[1]
    b = loaded.step(batch, h, r)[1]
    assert torch.equal(a, b)
    for k, v in small_model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_baseline_policies_shapes(small_model):
    batch = {k: v.numpy() for k, v in agent_batch(2, 2).items()}
    h = torch.zeros(8, 1)
    acts, _ = RandomPolicy(0).act(batch, h, torch.ones(8))
    assert acts.shape == (8, 5) and acts.dtype.name == "uint8"
    assert (acts.max(0) < [3, 3, 3, 2, 2]).all()
    acts, _ = IdlePolicy().act(batch, h, torch.ones(8))
    assert not acts.any()
    acts, new_h = NetPolicy(small_model).act(batch, small_model.initial_state(8), torch.ones(8))
    assert acts.shape == (8, 5) and new_h.shape == (8, 32)


def test_bundled_configs_load():
    for name in ("mock_small", "mock_tiny", "mock_curriculum", "warsim_small"):
        cfg = load_config(name)
        assert cfg.env.kind in ("mock", "warsim")


def test_overrides_and_validation():
    data = apply_overrides({"ppo": {"lr": 1.0}}, ["ppo.lr=3e-4", "env.kind=mock", "league.enabled=false"])
    assert data == {"ppo": {"lr": 3e-4}, "env": {"kind": "mock"}, "league": {"enabled": False}}
    cfg = config_from_dict(data)
    assert cfg.ppo.lr == 3e-4 and cfg.league.enabled is False
    with pytest.raises(ValueError):
        config_from_dict({"ppo": {"not_a_key": 1}})
    with pytest.raises(ValueError):
        config_from_dict({"ppo": {"rollout_steps": 10, "bptt_chunk": 4}})
    cfg = config_from_dict(json.loads(json.dumps({"curriculum": {"stages": [{"team_size": 3}]}})))
    assert cfg.curriculum.stages[0].team_size == 3

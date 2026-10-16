import dataclasses

import numpy as np
import torch

from warsim_rl.config import LeagueConfig, PPOConfig
from warsim_rl.env_factory import VecEnvAdapter
from warsim_rl.league import League
from warsim_rl.mock_env import MockVecEnv
from warsim_rl.ppo import MAPPO, ValueNormalizer, compute_gae
from warsim_rl.rollout import Collector, rescale_shaping


def test_gae_matches_hand_computation():
    rewards = torch.tensor([1.0, 0.0, 2.0])
    values = torch.tensor([0.5, 0.4, 0.3])
    dones = torch.tensor([0.0, 1.0, 0.0])
    adv, ret = compute_gae(rewards, values, dones, torch.tensor(1.0), gamma=0.9, lam=0.8)
    # t2: 2 + .9*1 - .3 = 2.6 ; t1 (terminal): -0.4 ; t0: 1 + .9*.4 - .5 + .9*.8*(-.4) = .572
    assert torch.allclose(adv, torch.tensor([0.572, -0.4, 2.6]), atol=1e-6)
    assert torch.allclose(ret, torch.tensor([1.072, 0.0, 2.9]), atol=1e-6)


def test_gae_broadcasts_over_envs_and_units():
    T, E, n = 5, 3, 2
    r, v = torch.randn(T, E, n), torch.randn(T, E, n)
    d = (torch.rand(T, E) < 0.3).float()[..., None].expand(T, E, n)
    last = torch.randn(E, n)
    adv, _ = compute_gae(r, v, d, last, 0.99, 0.95)
    for e in range(E):
        a1, _ = compute_gae(r[:, e, 1], v[:, e, 1], d[:, e, 1], last[e, 1], 0.99, 0.95)
        assert torch.allclose(adv[:, e, 1], a1)


def test_value_normalizer_roundtrip():
    vn = ValueNormalizer(beta=0.9)
    x = torch.tensor([0.0, 1.0])
    assert torch.allclose(vn.normalize(x), x)  # identity before any update
    vn.update(torch.randn(1000) * 3 + 5)
    mean, std = vn.stats()
    assert abs(mean - 5) < 0.5 and abs(std - 3) < 0.5
    assert torch.allclose(vn.denormalize(vn.normalize(x)), x, atol=1e-5)


def test_rescale_shaping_keeps_terminal():
    rewards = np.array([[1.3, -1.3], [0.2, -0.2]], dtype=np.float32)
    dones = np.array([True, False])
    winner = np.array([0, -1], dtype=np.int8)
    out = rescale_shaping(rewards, dones, winner, team_size=1, scale=0.0)
    assert np.allclose(out, [[1.0, -1.0], [0.0, 0.0]])


def _collect(model, ppo_cfg, n_envs=4):
    env = VecEnvAdapter(MockVecEnv(n_envs=n_envs, team_size=2, max_ticks=30, seed=0), n_envs, 2)
    league = League(LeagueConfig(enabled=False), "unused", seed=0)
    mappo = MAPPO(model, ppo_cfg)
    collector = Collector(env, model, mappo, league, ppo_cfg.rollout_steps)
    collector.collect()
    return mappo, collector.buffer


def test_buffer_minibatch_shapes(small_model):
    cfg = PPOConfig(rollout_steps=8, bptt_chunk=4, minibatches=2)
    _, buf = _collect(small_model, cfg)
    adv = torch.zeros_like(buf.rewards)
    mbs = list(buf.minibatches(2, 4, adv, adv))
    assert len(mbs) == 2
    total = sum(mb["h0"].shape[0] for mb in mbs)
    assert total == (8 // 4) * 4 * 2  # chunks * envs * learner units
    mb = mbs[0]
    L, BN = mb["resets"].shape
    assert L == 4 and mb["obs"]["tokens"].shape[:2] == (L, BN)
    assert mb["state"].shape == (L, BN, 4 * 10) and mb["actions"].shape == (L, BN, 5)


def test_ppo_loss_decreases_on_fixed_batch(small_model):
    torch.manual_seed(0)
    cfg = PPOConfig(
        rollout_steps=16, bptt_chunk=8, epochs=1, minibatches=1, ent_coef=0.0,
        ent_coef_final=0.0, lr=1e-3, lr_final=1e-3, value_norm_beta=0.5,
    )
    mappo, buf = _collect(small_model, cfg)
    first = mappo.update(buf, 0.0)
    for _ in range(25):
        last = mappo.update(buf, 0.0)
    assert last["loss"] < first["loss"]
    assert last["v_loss"] < first["v_loss"]
    assert last["pg_loss"] < first["pg_loss"] + 1e-6


def test_alive_mask_excludes_dead_units(small_model):
    cfg = PPOConfig(rollout_steps=8, bptt_chunk=4, epochs=1, minibatches=1)
    mappo, buf = _collect(small_model, cfg)
    buf.obs["alive"].zero_()
    before = [p.detach().clone() for p in small_model.parameters()]
    stats = mappo.update(buf, 0.0)
    assert stats["pg_loss"] == 0.0 and stats["v_loss"] == 0.0
    changed = any(not torch.equal(a, b) for a, b in zip(before, small_model.parameters()))
    # Adam with zero grads still leaves params unchanged
    assert not changed


def test_lr_and_entropy_schedules(small_model):
    cfg = dataclasses.replace(PPOConfig(), lr=1e-3, lr_final=1e-4, ent_coef=0.1, ent_coef_final=0.0)
    mappo = MAPPO(small_model, cfg)
    assert mappo.lr_sched.at(0.5) == 0.00055
    assert abs(mappo.ent_sched.at(1.0)) < 1e-12

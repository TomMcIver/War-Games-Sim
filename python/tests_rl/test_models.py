import torch

from conftest import agent_batch
from warsim_rl.constants import ACTION_DIMS
from warsim_rl.models import log_prob_entropy, sample_actions


def test_forward_shapes(small_model):
    batch = agent_batch(n_envs=3, team_size=2)
    B = batch["tokens"].shape[0]
    h = small_model.initial_state(B)
    logits, core, new_h = small_model.step(batch, h, torch.ones(B))
    assert [lg.shape for lg in logits] == [(B, d) for d in ACTION_DIMS]
    assert core.shape == (B, 32) and new_h.shape == (B, 32)
    v = small_model.value(batch["state"], batch["agent_idx"], core)
    assert v.shape == (B,)
    acts, logp = sample_actions(logits)
    assert acts.shape == (B, 5) and logp.shape == (B,)
    lp2, ent = log_prob_entropy(logits, acts)
    assert torch.allclose(logp, lp2, atol=1e-5) and torch.all(ent > 0)


def test_model_handles_different_team_sizes(small_model):
    for ts in (1, 3, 8):
        batch = agent_batch(n_envs=1, team_size=ts)
        B = batch["tokens"].shape[0]
        _, core, _ = small_model.step(batch, small_model.initial_state(B), torch.ones(B))
        assert small_model.value(batch["state"], batch["agent_idx"], core).shape == (B,)


def test_padded_tokens_do_not_change_outputs(small_model):
    torch.manual_seed(0)
    B, T = 5, 6
    tokens = torch.randn(B, T, 12)
    mask = torch.zeros(B, T)
    mask[:, :3] = 1
    lmap = torch.rand(B, 4, 9, 9)
    obs = {"tokens": tokens, "token_mask": mask, "local_map": lmap}
    garbage = tokens.clone()
    garbage[:, 3:] = torch.randn(B, T - 3, 12) * 100
    h = small_model.initial_state(B)
    reset = torch.zeros(B)
    out1 = small_model.step(obs, h, reset)[1]
    out2 = small_model.step({**obs, "tokens": garbage}, h, reset)[1]
    assert torch.allclose(out1, out2, atol=1e-5)
    # ... and removing the padding entirely gives the same result
    out3 = small_model.step(
        {"tokens": tokens[:, :3], "token_mask": mask[:, :3], "local_map": lmap}, h, reset
    )[1]
    assert torch.allclose(out1, out3, atol=1e-5)


def test_real_tokens_do_change_outputs(small_model):
    batch = agent_batch(n_envs=2, team_size=2)
    B = batch["tokens"].shape[0]
    h, reset = small_model.initial_state(B), torch.zeros(B)
    out1 = small_model.step(batch, h, reset)[1]
    changed = dict(batch, tokens=batch["tokens"].clone())
    changed["tokens"][:, 1, 3:5] += 0.5  # move the first ally
    assert not torch.allclose(out1, small_model.step(changed, h, reset)[1])


def test_gru_state_reset_on_dones(small_model):
    batch = agent_batch(n_envs=2, team_size=1)
    B = batch["tokens"].shape[0]
    zero = small_model.initial_state(B)
    junk = torch.randn(B, 32) * 3
    fresh = small_model.step(batch, zero, torch.zeros(B))[1]
    reset_out = small_model.step(batch, junk, torch.ones(B))[1]
    carried = small_model.step(batch, junk, torch.zeros(B))[1]
    assert torch.allclose(fresh, reset_out, atol=1e-6)
    assert not torch.allclose(fresh, carried)


def test_forward_seq_matches_stepwise(small_model):
    L, n_env = 4, 2
    seq = [agent_batch(n_envs=n_env, team_size=1, seed=s) for s in range(L)]
    B = seq[0]["tokens"].shape[0]
    resets = torch.zeros(L, B)
    resets[2, 0] = 1.0
    h0 = torch.randn(B, 32)
    stacked = {k: torch.stack([s[k] for s in seq]) for k in ("tokens", "token_mask", "local_map", "alive")}
    logits_seq, core_seq = small_model.forward_seq(stacked, h0, resets)
    h = h0
    for t in range(L):
        logits, core, h = small_model.step(seq[t], h, resets[t])
        assert torch.allclose(core, core_seq[t], atol=1e-5)
        assert torch.allclose(logits[0], logits_seq[0][t], atol=1e-5)


def test_dead_units_only_pick_noop(small_model):
    batch = agent_batch(n_envs=2, team_size=2)
    batch["alive"][:3] = 0.0
    B = batch["tokens"].shape[0]
    logits, _, _ = small_model.step(batch, small_model.initial_state(B), torch.ones(B))
    for _ in range(5):
        acts, _ = sample_actions(logits)
        assert torch.all(acts[:3] == 0)
    _, ent = log_prob_entropy(logits, acts)
    assert torch.allclose(ent[:3], torch.zeros(3), atol=1e-6)

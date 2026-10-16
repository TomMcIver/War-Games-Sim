import math

import numpy as np
import pytest

from warsim_rl.config import LeagueConfig
from warsim_rl.elo import EloTable, expected_score, performance_rating, update_elo
from warsim_rl.league import LATEST, LEARNER, League, pfsp_weights
from warsim_rl.policies import RandomPolicy


def test_expected_score_and_update():
    assert expected_score(1000, 1000) == 0.5
    assert math.isclose(expected_score(1400, 1000), 1 / (1 + 10 ** -1))
    a, b = update_elo(1000, 1000, 1.0, k=32)
    assert (a, b) == (1016, 984)
    a, b = update_elo(1200, 1000, 0.5, k=16)
    assert a < 1200 and b > 1000 and math.isclose(a + b, 2200)
    with pytest.raises(ValueError):
        update_elo(1000, 1000, 1.5)


def test_performance_rating():
    assert math.isclose(performance_rating(1000, 5, 10), 1000)
    assert math.isclose(performance_rating(1000, 10 * 10 / 11, 10), 1000 + 400)
    assert math.isfinite(performance_rating(1000, 10, 10))


def test_elo_table_anchor_fixed():
    t = EloTable(1000, 16, anchors=["random"])
    for _ in range(10):
        t.record("learner", "random", 1.0)
    assert t.get("random") == 1000 and t.get("learner") > 1000
    t2 = EloTable()
    t2.load_state_dict(t.state_dict())
    assert t2.ratings == t.ratings and t2.anchors == t.anchors


def test_pfsp_weights():
    w = pfsp_weights(np.array([0.9, 0.5, 0.1]), "hard", 2.0)
    assert np.isclose(w.sum(), 1) and w[2] > w[1] > w[0]
    w = pfsp_weights(np.array([0.9, 0.5, 0.1]), "even", 1.0)
    assert w[1] > w[0] and np.isclose(w[0], w[2])
    assert np.allclose(pfsp_weights(np.array([0.2, 0.7]), "uniform", 1.0), 0.5)


def test_league_sampling_probabilities(tmp_path):
    league = League(LeagueConfig(p_latest=0.7), tmp_path, seed=0)
    draws = [league.sample_opponent() for _ in range(4000)]
    frac_latest = draws.count(LATEST) / len(draws)
    assert abs(frac_latest - 0.7) < 0.03
    assert set(draws) == {LATEST, "random"}
    sides = [league.sample_learner_team() for _ in range(2000)]
    assert abs(np.mean(sides) - 0.5) < 0.05


def test_league_prefers_hard_opponents(tmp_path, small_model):
    league = League(LeagueConfig(p_latest=0.0, include_random=True, pfsp_decay=1.0), tmp_path, seed=0)
    snap = league.add_snapshot(small_model, 10)
    for _ in range(30):
        league.record("random", 1.0)  # learner crushes random
        league.record(snap, 0.0)  # but loses to the snapshot
    draws = [league.sample_opponent() for _ in range(1000)]
    assert draws.count(snap) > 0.9 * len(draws)
    assert league.elo.get(snap) > league.elo.get(LEARNER)
    assert league.elo.get("random") == 1000.0  # anchor


def test_league_snapshot_pool_and_state(tmp_path, small_model):
    league = League(LeagueConfig(pool_size=2), tmp_path, seed=0)
    ids = [league.add_snapshot(small_model, u) for u in (5, 10, 15)]
    snaps = [e for e in league.entries.values() if e.path]
    assert [e.id for e in snaps] == ids[1:]  # oldest pruned
    assert "random" in league.entries
    assert league.latest_snapshot_path().endswith("u000015.pt")
    policy = league.get_policy(ids[-1])
    assert policy.hidden_size == small_model.hidden_size
    assert isinstance(league.get_policy("random"), RandomPolicy)
    league.record(ids[-1], 1.0)
    league.record(LATEST, 0.5)
    other = League(LeagueConfig(pool_size=2), tmp_path, seed=99)
    other.load_state_dict(league.state_dict())
    assert list(other.entries) == list(league.entries)
    assert other.elo.ratings == league.elo.ratings
    assert other.sample_opponent() == league.sample_opponent()  # rng restored
    league.write_json(tmp_path / "league.json")
    assert (tmp_path / "league.json").read_text().startswith("{")

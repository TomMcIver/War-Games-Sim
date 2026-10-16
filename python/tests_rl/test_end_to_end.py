"""End-to-end smoke tests: tiny training runs that finish in seconds."""

import csv
import json

from warsim_rl import evaluate, train
from warsim_rl.config import StageConfig, load_config
from warsim_rl.env_factory import make_vec_env
from warsim_rl.evaluate import play_match
from warsim_rl.policies import IdlePolicy, RandomPolicy
from warsim_rl.trainer import Trainer


def tiny(tmp_path, **changes):
    cfg = load_config("mock_tiny")
    cfg.out_dir = str(tmp_path)
    for k, v in changes.items():
        setattr(cfg, k, v)
    return cfg


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_train_cli_checkpoint_resume_and_evaluate(tmp_path, capsys):
    out = str(tmp_path)
    row = train.main(["--config", "mock_tiny", "--out-dir", out, "--name", "smoke"])
    run = tmp_path / "smoke"
    assert row["update"] == 3
    rows = read_rows(run / "metrics.csv")
    assert len(rows) == 3 and rows[-1]["global_step"] == str(3 * 4 * 8)
    assert (run / "checkpoints" / "latest.pt").is_file()
    assert sorted(p.name for p in (run / "pool").glob("u*.pt"))[-1] == "u000003.pt"
    league = json.loads((run / "league.json").read_text())
    assert league["anchor"] == "random"

    # Resume: continues from update 3 to 5 and appends to the same CSV.
    row = train.main(["--config", "mock_tiny", "--out-dir", out, "--name", "smoke",
                      "--updates", "5", "--resume", "auto"])
    assert row["update"] == 5
    assert [r["update"] for r in read_rows(run / "metrics.csv")] == ["1", "2", "3", "4", "5"]

    ckpt = str(run / "checkpoints" / "latest.pt")
    for opp in ("random", "idle", "latest"):
        rep = evaluate.main(["--checkpoint", ckpt, "--opponent", opp, "--episodes", "4"])
        assert rep["episodes"] == 4 and rep["wins"] + rep["draws"] + rep["losses"] == 4
        assert "performance_elo" in rep
    rep = evaluate.main(["--checkpoint", ckpt, "--opponent", "checkpoint",
                         "--opponent-checkpoint", str(run / "pool" / "u000002.pt"), "--episodes", "3"])
    assert rep["episodes"] == 3
    capsys.readouterr()


def test_same_seed_same_results(tmp_path):
    rows = []
    for name in ("a", "b"):
        t = Trainer(tiny(tmp_path, name=name, total_updates=2), quiet=True)
        rows.append(t.train())
        t.close()
    for key in ("loss", "v_loss", "entropy", "eval_win_rate_random", "learner_elo"):
        assert rows[0][key] == rows[1][key], key


def test_curriculum_promotion_rebuilds_env(tmp_path):
    cfg = tiny(tmp_path, name="cur", total_updates=4, eval_every=1)
    cfg.curriculum.stages = [
        StageConfig(team_size=1, map_kind="open", promote_winrate=0.0, min_updates=1),
        StageConfig(team_size=3, map_kind="cover", promote_winrate=2.0, min_updates=1),
    ]
    t = Trainer(cfg, quiet=True)
    t.train()
    assert t.curriculum.index == 1 and t.env.n_units == 6
    rows = read_rows(tmp_path / "cur" / "metrics.csv")
    assert rows[0]["stage"] == "1" and rows[0]["team_size"] == "3"  # promoted at update 1
    assert rows[-1]["team_size"] == "3"
    t.close()


def test_play_match_random_vs_idle_and_quota():
    cfg = load_config("mock_tiny").env
    cfg.max_ticks = 60
    env = make_vec_env(cfg, seed=0, n_envs=4)
    res = play_match(RandomPolicy(0), IdlePolicy(), env, episodes=10)
    assert res.episodes == 10
    assert res.losses == 0  # idle never shoots, so it can never win

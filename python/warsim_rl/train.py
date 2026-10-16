"""Training CLI.

    python -m warsim_rl.train --config mock_small
    python -m warsim_rl.train --config warsim_rl/configs/mock_small.json \
        --set ppo.lr=3e-4 --set env.n_envs=16 --name exp1 --resume auto
"""

from __future__ import annotations

import argparse
from typing import Optional

from .config import load_config
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train warsim_rl (MAPPO + self-play league)")
    p.add_argument("--config", default="mock_small", help="bundled config name or JSON/YAML path")
    p.add_argument("--set", action="append", default=[], help="override, e.g. ppo.lr=3e-4")
    p.add_argument("--name", help="run name (default: config's name)")
    p.add_argument("--seed", type=int)
    p.add_argument("--updates", type=int, help="total PPO updates")
    p.add_argument("--out-dir", help="output root (default: runs)")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", help="'auto' (runs/<name>/checkpoints/latest.pt) or a path")
    return p


def main(argv: Optional[list] = None) -> dict:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.updates is not None:
        cfg.total_updates = args.updates
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.deterministic:
        cfg.deterministic = True
    trainer = Trainer(cfg, resume=args.resume)
    try:
        return trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

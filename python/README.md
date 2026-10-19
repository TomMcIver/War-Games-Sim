# warsim_rl: the learning AI

This package trains the War-Games-Sim soldiers with **MAPPO plus a self-play
league**, a small CPU-sized version of the OpenAI Five / AlphaStar recipe.
There are no scripted tactics. One shared neural network controls every unit
and learns only from reward. The trainer runs against the real Rust sim
(`warsim`) or against a pure-numpy stand-in (`MockVecEnv`) that has the same
API, so all of the RL code can be developed and tested without Rust.

All Python runs in Podman (host Python is not used). Build the image once with
`bash infra/build-image.sh`. Then, from the repo root:

```bash
# tests (about 5-10 s); the real-sim tests are skipped when warsim isn't installed
bash infra/run.sh bash -c "cd python && pip install -e . -q && pytest tests_rl -q"

# train on the mock env (CSV and checkpoints go to python/runs/<name>/)
bash infra/run.sh bash -c "cd python && pip install -e . -q && python -m warsim_rl.train --config mock_small"

# evaluate a checkpoint against the random baseline
bash infra/run.sh bash -c "cd python && pip install -e . -q && python -m warsim_rl.evaluate \
    --checkpoint runs/mock_small/checkpoints/latest.pt --opponent random --episodes 200"
```

### Against the real sim

Build the `warsim` wheel once. It goes into `target/wheels`, which is gitignored.
After that, install it in every container run:

```bash
bash infra/run.sh bash -c "cd crates/sim-py && maturin build --release -o /work/target/wheels"
bash infra/run.sh bash -c "pip install -q /work/target/wheels/warsim-*.whl && cd python && pip install -e . -q \
    && pytest tests_rl -q && python -m warsim_rl.train --config warsim_small"
```

## Layout

| module | what it does |
|---|---|
| `mock_env.py` | `MockVecEnv`: numpy 2-team toy shooter with the `warsim.VecEnv` API, shapes, dtypes and auto-reset |
| `env_factory.py` | `make_vec_env(cfg)` picks `env.kind: mock \| warsim`; `VecEnvAdapter` normalises obs, dones and infos |
| `models.py` | `ActorCritic`: the shared actor and the centralised critic |
| `ppo.py` | GAE, `ValueNormalizer`, `RolloutBuffer` `[T, E, n]`, `MAPPO.update` |
| `rollout.py` | `Collector`: batched learner and opponent acting, side swap, Python-side shaping anneal |
| `league.py`, `elo.py` | opponent pool, PFSP sampling, snapshots, Elo |
| `curriculum.py` | `(team_size, map_kind, max_ticks)` stages, promoted on a win-rate threshold |
| `trainer.py`, `train.py` | training loop and CLI (CSV + stdout logging, checkpoint/resume) |
| `evaluate.py` | `play_match` and the evaluation CLI (W/D/L + performance Elo) |
| `policies.py` | `NetPolicy`, `RandomPolicy`, `IdlePolicy` |
| `configs/*.json` | `mock_tiny` (tests), `mock_small`, `mock_curriculum`, `warsim_small` |

## Model

- **Actor.** It sees only its own unit's observation: `tokens [T,12]`, `token_mask`, and `local_map [4,9,9]`.
  - Tokens: linear embed (d=64), then 2 pre-norm masked self-attention layers (4 heads).
  - Pooling: masked attention-pool, with the self token as the query.
  - Map: a small CNN (`4→16→32`, stride 2) followed by a linear layer.
  - Head: concat, then a 2-layer MLP (128), then a `GRUCell(128)`. The hidden state is zeroed at episode starts.
  - Output: 5 categorical heads `[3,3,3,2,2]`. The joint log-prob and entropy are the sum over heads.
  - Padded tokens are excluded as attention keys and zeroed on output, so they cannot affect anything (this is tested).
  - Dead units can only pick action 0, and they are left out of every loss term via `alive`.
- **Critic (CTDE).** `state [U*10]` is treated as a set of U unit entities, each tagged `is_self`/`is_teammate`.
  - It runs attention, then an attention-pool keyed on the agent's own entity, plus the actor's GRU output (detached). That gives one value per unit.
  - Because the critic works on a set, it handles any team size and either side without changes. Curriculum stages can change `team_size` mid-run.
- Orthogonal init throughout: action heads use gain 0.01 and the value head uses gain 1.

## MAPPO

- Rollouts are `[steps, envs, learner_units]`.
- Advantages: GAE(γ, λ), with `dones` cutting the bootstrap.
- Policy loss: clipped PPO, with advantages normalised per minibatch over alive agents.
- Value loss: clipped, on returns normalised by a debiased EMA (MAPPO ValueNorm).
- Schedules: entropy bonus annealed linearly from `ent_coef` to `ent_coef_final`; LR decays linearly from `lr` to `lr_final`. Gradients are clipped to `max_grad_norm`. `target_kl` early stopping is optional.
- Recurrent minibatches: each minibatch is a set of `(chunk, env)` sequences of length `bptt_chunk` (truncated BPTT). Each sequence starts from the hidden state stored during the rollout, with resets applied inside the chunk.
- Shaping anneal: the shaping part of the reward is rescaled on the Python side, `r' = terminal + s·(r − terminal)`. `s` goes linearly from `env.shaping_scale` to `env.shaping_scale_final`. This works with any env that follows INTERFACE.md and needs no env rebuild. The real bindings fix `shaping_scale` at construction time.

## Self-play league

- The learner plays one team and the opponent plays the other.
- **Opponent choice.** Each env draws a new opponent at every episode start:
  - `latest` (the current weights) with probability `p_latest`. Its forward pass is batched together with the learner's.
  - Otherwise, a frozen pool member: past snapshots plus `random` (and optionally `idle`). Pool members are weighted by PFSP. In `hard` mode the weight is `(1 − learner win-rate)^p`, with exponentially decayed counts.
- **Side swap.** With probability `side_swap_prob` the learner plays team 1, so the policy learns both positions.
- **Snapshots.** Every `snapshot_every` updates the current weights are saved to `runs/<name>/pool/uXXXXXX.pt`. Only the newest `pool_size` snapshots stay in the pool.
- **Elo.** Every pool episode updates Elo between the learner and that opponent. `random` is the fixed anchor at 1000. A new snapshot starts with the learner's current rating. Ratings are written to `runs/<name>/league.json`.

## Training CLI

```
python -m warsim_rl.train --config mock_small [--set ppo.lr=3e-4 --set env.n_envs=16 ...]
       [--name NAME] [--seed N] [--updates N] [--out-dir DIR] [--deterministic] [--resume auto|PATH]
```

- **Config.** `--config` takes a bundled config name or a JSON path (YAML works if pyyaml is installed). `--set` overrides any nested key, and the value is parsed as JSON.
- **Output.** Each update writes one row to `runs/<name>/metrics.csv` and one line to stdout. Columns:
  - training: rolling win/draw rate and score vs the pool / vs latest, episode return and length
  - evaluation: `eval_*_random`, a periodic evaluation vs `RandomPolicy`
  - league: learner Elo, pool size
  - optimiser: pg/value loss, entropy, approx-KL, clip fraction, grad norm, explained variance, lr, entropy coefficient, shaping scale
  - throughput: env-steps/sec and agent-steps/sec
- **Checkpoints.** Written to `runs/<name>/checkpoints/latest.pt` every `checkpoint_every` updates. They hold the model, optimiser, value-norm, league, curriculum, counters and torch RNG. `--resume auto` picks the run up from there.
- **Curriculum.** Set `curriculum.stages: [{team_size, map_kind, scenario, max_ticks, promote_winrate, min_updates, min_episodes}]`. The promotion metric is win rate in `eval_random` (the default) or `league`. A stage needs both minimum exposure and wins to promote. On promotion the env is rebuilt; the model is size-agnostic. The bundled real-sim config starts with close duels, then cover, doors, 2v2 skirmishes, and full maps.
- **Action availability.** The actor uses simulator-provided shoot/interact availability masks for physically impossible actions. The same masks enter rollout sampling and PPO updates. Dead-unit actions are masked on every head. Interaction resolves before movement so the mask is accurate for the action tick.

## Evaluation CLI

```
python -m warsim_rl.evaluate --checkpoint X --opponent random|idle|checkpoint|latest
       [--opponent-checkpoint Y] [--episodes N] [--deterministic] [--no-swap]
       [--config C] [--set k=v] [--env-kind mock|warsim] [--team-size N] [--map-kind K] [--scenario S] [--max-ticks T]
```

- **Output.** JSON with wins, draws, losses, rates, mean episode length, the opponent's Elo (from the run's `league.json`, or 1000 for random) and the checkpoint's performance Elo.
- **Sides.** Sides alternate across envs.
- **Fair counting.** Each env plays a fixed quota of episodes, so short episodes are not over-counted.
- **Env settings.** These come from the checkpoint, including its curriculum stage, unless overridden.

For a repeatable real-simulator random baseline without a checkpoint, run
`python -m warsim_rl.baseline --episodes 64 --opponent random`. The JSON
includes termination reasons, contact ticks, shots, hits, damage, deaths, and
ending ammo as well as the match outcome. `--opponent idle` isolates whether
random actions can find and hit an unmoving enemy.

For a fixed real-simulator comparison after checkpoint selection, run:

```bash
python -m warsim_rl.heldout \
  --checkpoint runs/NAME/pool/u000050.pt \
  --checkpoint runs/NAME/checkpoints/latest.pt \
  --opponents random idle frozen \
  --frozen-checkpoint runs/NAME/pool/u000050.pt \
  --out runs/NAME/heldout.json
```

The report records every seed, scenario, checkpoint, and opponent, plus
aggregate outcomes, win-rate intervals, contact, timeout, hit rate, damage,
and ammo. Matches use equal quotas on both sides. Keep the fixed seeds out of
training and checkpoint selection; interpreting improvements still requires
a sufficiently trained real-simulator checkpoint.

## Mock env (`MockVecEnv`)

- **Game.**
  - Arena: 12×12, continuous positions, random pillar walls. `map_kind`: `open` / `arena` / `cover`.
  - Movement: units move, strafe and turn (22.5° steps).
  - Shooting: hitscan at the nearest enemy inside a ±15° cone, within range 7, with clear line of sight. Shots cost ammo and have a cooldown; 3 hits kill.
  - Vision: team-shared, range 8, with line of sight.
- **Rewards.** Team-shared +1/−1 for a wipe-out. A timeout is a draw (0). Shaping: 0.2·(damage dealt − damage taken) + 0.2·(kills − deaths).
- **Interface.** Same obs keys, shapes, dtypes, token ordering (self, allies, visible enemies, padding), auto-reset, `infos` and determinism as INTERFACE.md.
- **Simplifications.** Single floor, no doors or stairs (`interact` does nothing), local map not fogged, last-seen-age always 0.

## Learning on the mock env

Run setup: the bundled `mock_small` config.

- **Env:** 2v2, `arena`, 32 envs, max 120 ticks, shaping annealed from 1.0 to 0.25.
- **Training:** 150 updates × 64 steps = 307k env-steps.
- **Model:** d_model 64, GRU 128.
- **League:** `p_latest` 0.5, a snapshot every 10 updates, side swap 0.5, seed 0.
- **Hardware:** 4 CPU threads in Podman. Wall time was about 15 minutes, but other container jobs were competing for the CPU.

```bash
bash infra/run.sh bash -c "cd python && pip install -e . -q && python -m warsim_rl.train --config mock_small --name mock_small_s0"
```

Periodic evaluation vs `RandomPolicy`: 64 episodes each, alternating sides, sampled actions. These are the `eval_win_rate_random` / `eval_loss_rate_random` columns of `metrics.csv`. The rest are draws, i.e. timeouts.

| update | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win | 0.03 | 0.16 | 0.17 | 0.55 | 0.64 | 0.39 | 0.31 | 0.53 | 0.66 | 0.81 | 0.86 | 0.84 | 0.78 | 0.91 | 0.89 |
| loss | 0.02 | 0 | 0 | 0 | 0 | 0 | 0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Other signals over the same run:

- Mean episode length fell from 119 to 41 ticks: the policy learned to close in and kill rather than time out.
- League Elo rose from 1000 to 1482, with random anchored at 1000.
- Policy entropy fell from 4.68 to 3.33.
- Explained variance rose from −0.2 to 0.6.

Final evaluation, 200 episodes each:

```bash
bash infra/run.sh bash -c "cd python && pip install -e . -q && python -m warsim_rl.evaluate \
    --checkpoint runs/mock_small_s0/checkpoints/latest.pt --opponent random --episodes 200"
```

| checkpoint | opponent | W / D / L | win rate | perf. Elo |
|---|---|---|---|---|
| final (u150) | random | 172 / 28 / 0 | **0.86** | 1449 |
| final (u150), `--deterministic` | random | 180 / 20 / 0 | **0.90** | 1512 |
| final (u150) | idle | 174 / 26 / 0 | 0.87 | 1463 |
| final (u150) | snapshot u50 | 162 / 21 / 17 | 0.81 | 1368 |
| final (u150) | itself (`latest`) | 86 / 24 / 90 | 0.43 | 1475 |
| early snapshot u10 | random | 7 / 190 / 3 | 0.035 | 1007 |

The `itself` row is a sanity check: the same weights against each other come out at about 50/50.

**Real sim.** `--config warsim_small` starts on the close 1v1 duel stage (32 envs × 128 steps), then promotes by measured wins and completed episodes. Learning curves on the revised real-sim curriculum have not been measured yet.

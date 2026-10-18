# Timed real-simulator learning experiment

The local run `runs/timed_30m_s1/` trains a fresh seed-1 policy for about
30 minutes of wall-clock time with the bundled `warsim_small` curriculum.
It uses a 2,000-update schedule so learning rate, entropy, and shaping decay
are consistent throughout this shorter observation window. Training happens
in chunks of 50 PPO updates; the wall-clock budget is checked between chunks.

At update 0 and after every chunk, the run saves a checkpoint under
`checkpoints/history/` and plays 32 matches on each of two fixed validation
seeds (31001, 42002) against a seeded random policy. Sides are balanced.
The fixed duel benchmark is repeated at every checkpoint; once the curriculum
promotes, the current stage is also evaluated. These validation results do
not enter gradients or curriculum promotion. Promotion uses its own simulator
seed (104730 for seed 1). A separate final held-out suite reserves seeds
71011, 82021, 93031, and 104041 for later checkpoint comparison.

The chart compares:

- rolling training wins against league opponents;
- promotion evaluation against random on the current stage;
- fixed validation against random on the current stage and on the original duel;
- duel timeout rate, learner shot hit rate, and episode length.

These curves answer different questions. The league pool changes during
training, so its win rate is not numerically comparable with the fixed
validation win rate. The fixed duel curve can show retention or forgetting
after a stage transition. A plateau should be judged from several validation
points and their match-count uncertainty, not one flat interval.

Reproduction:

```bash
PYTHONPATH=python python -m warsim_rl.timed_experiment \
  --minutes 30 --seed 1 --name timed_30m_s1 \
  --chunk-updates 50 --validation-episodes-per-seed 32
pip install -e 'python[plot]'
PYTHONPATH=python python -m warsim_rl.plot_learning runs/timed_30m_s1
```

The run writes `metrics.csv`, `learning_curve.csv`, `experiment.json`,
`learning_curve.png`, and `learning_curve.svg` under its run directory.
The shareable [learning-curve graph](experiments/timed_30m_s1.png) and
[CSV data](experiments/timed_30m_s1.csv) are copied into this repository.

## Observed result

The seed-1 run completed 352 updates in 30.31 minutes. The curriculum
promoted from duel to cover at update 200 and did not reach door. Each fixed
validation point contains 64 matches, balanced across player sides.

| Update | Stage | Stage wins / draws / losses | Fixed duel wins / draws / losses |
| ---: | --- | ---: | ---: |
| 0 | duel | 10 / 46 / 8 | 10 / 46 / 8 |
| 150 | duel | 41 / 21 / 2 | 41 / 21 / 2 |
| 200 | cover | 1 / 63 / 0 | 59 / 5 / 0 |
| 250 | cover | 6 / 56 / 2 | 60 / 3 / 1 |
| 300 | cover | 17 / 45 / 2 | 56 / 6 / 2 |
| 350 | cover | 24 / 39 / 1 | 41 / 22 / 1 |
| 352 | cover | 31 / 32 / 1 | 43 / 20 / 1 |

Duel wins rose from 10/64 untrained to 59/64 at promotion. The new cover
scenario initially produced 1/64 wins, then recovered to 31/64 by the
timer's end. The fixed duel result fell from its 60/64 peak to 43/64 while
cover was improving. This is evidence of forgetting, though each validation
estimate has sampling uncertainty. The adjacent updates 350–352 also vary
substantially on the same seeds, so the final cover value should not be
treated as a stable level. No plateau is established by this run.

The run was resumed once from the update-300 checkpoint after discovering
that the league's 8-model cache repeatedly reloaded a pool of up to 30
opponents. The resume raised cache capacity to 32, retaining the model,
optimizer, curriculum state, random state, and configured learning schedule.
This repaired evaluation throughput without changing the training objective.

The duel and cover scenarios use fixed geometry, so even held-out random
seeds mainly test action and opponent randomness there. They cannot establish
generalization to new maps or opponents.

## One-time held-out check

After training, checkpoints from updates 200 and 352 were compared against
the seeded random policy on reserved seeds 71011, 82021, 93031, and 104041.
Each scenario and checkpoint had 64 matches, with sides balanced. The
[machine-readable results](experiments/timed_30m_s1_heldout.json) contain the
full telemetry.

| Scenario | Update 200 wins / draws / losses | Update 352 wins / draws / losses |
| --- | ---: | ---: |
| Duel | 63 / 1 / 0 | 44 / 18 / 2 |
| Cover | 0 / 63 / 1 | 34 / 30 / 0 |
| Arena | 2 / 62 / 0 | 1 / 63 / 0 |

The reserved seeds confirm the cover gain and duel regression. Arena remained
almost entirely timeouts, so this experiment does not show transfer to that
scenario. A next experiment should protect earlier-stage performance during
cover training and evaluate more map layouts and opponent policies.

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

## Rehearsal of earlier stages

The forgetting above motivated `curriculum.rehearsal_frac`. After a promotion,
that share of the training environments (here 25%, 8 of 32) keeps playing
earlier stages of the same team size; the rest play the current stage. Only
current-stage episodes count toward promotion. The run `timed_30m_s1_rehearse25`
repeats the seed-1, 30-minute experiment with `--rehearsal-frac 0.25`:

```bash
PYTHONPATH=python python -m warsim_rl.timed_experiment \
  --minutes 30 --seed 1 --name timed_30m_s1_rehearse25 \
  --chunk-updates 50 --validation-episodes-per-seed 32 --rehearsal-frac 0.25
```

It completed 446 updates in 30.92 minutes (the first run: 352) and promoted from
duel to cover at update 200, as before.
Validation uses the same fixed seeds, 64 matches per point. The
[graph](experiments/timed_30m_s1_rehearse25.png) and
[CSV](experiments/timed_30m_s1_rehearse25.csv) are in this repository.

| Update | Cover wins / draws / losses | Fixed duel wins / draws / losses |
| ---: | ---: | ---: |
| 200 | 0 / 64 / 0 | 58 / 4 / 2 |
| 250 | 5 / 59 / 0 | 59 / 2 / 3 |
| 300 | 6 / 58 / 0 | 59 / 5 / 0 |
| 350 | 7 / 56 / 1 | 61 / 2 / 1 |
| 400 | 9 / 54 / 1 | 60 / 3 / 1 |
| 446 | 23 / 41 / 0 | 61 / 2 / 1 |

Duel retention is fixed: the duel win count stayed at 58–61 of 64 after
promotion, where the first run fell to 41–43 of 64. The cost is slower cover
learning. At update 350 cover stood at 7/64 against 24/64 without rehearsal,
and the final 23/64 at update 446 is below the first run's 31/64 at update
352 despite 94 more updates. Cover wins jumped from 9 to 23 over the last 46
updates, so the end point is unsettled, as in the first run.

This is one seed with 64 matches per point, so treat the retention gain as
clear (about 60 versus 43 of 64) and the cover slowdown as likely but not
established. Not yet run: the reserved held-out suite on these checkpoints, a
second seed, other rehearsal fractions (for example 0.1 or an annealed
share), and door or skirmish stages, where rehearsal splits across more than
one earlier stage.

## Second seed and a smaller rehearsal share

Three more 30-minute runs used the same setup: seed 2 without rehearsal,
seed 2 with `--rehearsal-frac 0.25`, and seed 1 with `--rehearsal-frac 0.10`.
Wall-clock budgets mean update counts differ, so rows are matched by update.
Entries are wins / draws / losses over 64 matches versus random; "duel" is the
fixed duel anchor. Graphs and CSVs are in `docs/experiments/`.

| Run | Update | Stage | Stage result | Duel anchor |
| --- | ---: | --- | ---: | ---: |
| seed 1, none | 350 | cover | 24 / 39 / 1 | 41 / 22 / 1 |
| seed 1, 10% | 350 | cover | 5 / 59 / 0 | 57 / 7 / 0 |
| seed 1, 25% | 350 | cover | 7 / 56 / 1 | 61 / 2 / 1 |
| seed 2, none | 400 | door | 16 / 47 / 1 | 63 / 1 / 0 |
| seed 2, 25% | 400 | door | 46 / 18 / 0 | 59 / 4 / 1 |

Seed 1 final points: 10% reached cover 19/64 at update 373 with the duel
anchor at 58/64; 25% reached 23/64 at update 446 with 61/64.

What this supports:

- Seed 1 forgot duel without rehearsal (peak 60/64, then 41-43/64); both
  rehearsal fractions largely prevented that (10%: 46-60/64, 25%: 58-61/64).
- Seed 2 did not forget without rehearsal (duel anchor 54-63/64 throughout),
  so there was little for rehearsal to fix, and 25% rehearsal showed no cost
  there. At update 400 it was ahead on door (46/64 against 16/64), but the
  control's door results swing widely (5/64 at update 411).
- The cover slowdown seen on seed 1 did not appear on seed 2. With two seeds,
  neither the forgetting nor the slowdown is established as general.

Still not run: the reserved held-out suite on any rehearsal checkpoint, more
seeds, annealed rehearsal, and skirmish stages. The seed-1 baseline directory
`runs/timed_30m_s1` was later overwritten by an aborted resume; the committed
`docs/experiments/timed_30m_s1.csv` holds the run reported above.

## Held-out check of the final checkpoints

The five final checkpoints were played once against the seeded random policy
on the reserved seeds (71011, 82021, 93031, 104041), 64 matches per scenario,
sides balanced. Wins / draws / losses; full telemetry is in
[rehearsal_heldout.json](experiments/rehearsal_heldout.json).

| Run (final update) | Duel | Cover | Door |
| --- | ---: | ---: | ---: |
| seed 1, none (352) | 46 / 18 / 0 | 37 / 27 / 0 | 2 / 62 / 0 |
| seed 1, 10% (373) | 61 / 3 / 0 | 31 / 32 / 1 | 6 / 58 / 0 |
| seed 1, 25% (446) | 62 / 2 / 0 | 33 / 31 / 0 | 3 / 61 / 0 |
| seed 2, none (411) | 58 / 6 / 0 | 37 / 27 / 0 | 4 / 60 / 0 |
| seed 2, 25% (400) | 61 / 3 / 0 | 46 / 17 / 1 | 46 / 17 / 1 |

- Seed 1: rehearsal kept duel at 61-62/64 against 46/64 without it, and
  the final cover results are similar (31-37/64), so the in-training cover
  gap at update 350 had mostly closed by these checkpoints.
- Seed 2: 25% rehearsal was better on cover (46 against 37) and far better
  on door (46 against 4), although that run had the same promotion schedule
  luck as before; one seed does not separate rehearsal from run-to-run noise.
- No scenario here shows rehearsal clearly hurting. Door and arena
  generalization beyond fixed geometry remain untested.

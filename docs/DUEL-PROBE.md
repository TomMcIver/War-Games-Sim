# Short real-simulator training probe

On 2026-10-19 a local `warsim_small` run trained through update 20 on the
close 1v1 `duel` stage (seed 0, 32 environments, 128 rollout steps, 16-match
promotion evaluation every five updates). The run is in
`runs/curriculum_probe_s0/`. It resumed from update 10 and retained both
checkpoints. This is a probe, not evidence of general combat competence.

```bash
python -m warsim_rl.train --config warsim_small --name curriculum_probe_s0 \
  --updates 10 --set eval_every=5 --set eval_episodes=16
python -m warsim_rl.train --config warsim_small --name curriculum_probe_s0 \
  --updates 20 --set eval_every=5 --set eval_episodes=16 --resume auto
python -m warsim_rl.heldout \
  --checkpoint runs/curriculum_probe_s0/checkpoints/u000010.pt \
  --checkpoint runs/curriculum_probe_s0/checkpoints/latest.pt \
  --opponents random idle frozen \
  --frozen-checkpoint runs/curriculum_probe_s0/checkpoints/u000010.pt \
  --scenarios duel --seeds 71011 82021 --episodes-per-seed 16 --n-envs 8 \
  --out runs/curriculum_probe_s0/heldout_duel_probe.json
```

The fixed comparison had 32 matches per checkpoint and opponent, split
equally between sides. The win counts were:

| Checkpoint | Random | Idle | Frozen update 10 |
|---|---:|---:|---:|
| Update 10 | 7 W / 19 D / 6 L | 12 W / 20 D / 0 L | 3 W / 24 D / 5 L |
| Update 20 | 12 W / 15 D / 5 L | 12 W / 20 D / 0 L | 8 W / 16 D / 8 L |

Against random, the learner's shot hit rate rose from 8.0% to 10.8%, and
timeouts fell from 19/32 to 15/32. This suggests progress within the duel
stage, but 32 matches per row give wide uncertainty. The full held-out suite
has not been run, and the policy has not advanced to cover, doors, team
fights, or procedural maps. Duel geometry and spawn positions are fixed, so
these seeds vary combat randomness and episode streams, not the map layout.
Do not infer broader learning from this probe.

The detailed machine-readable report in the ignored run directory includes
per-seed counts, 95% win-rate intervals, contact, learner shots/hits, damage,
deaths, survivors, and ending ammo. The main training metrics are in
`runs/curriculum_probe_s0/metrics.csv`.

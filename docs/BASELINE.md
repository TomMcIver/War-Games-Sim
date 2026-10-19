# Real-simulator combat baseline

These are measurements from the Rust simulator on 2026-10-19, before any
combat-rule or training-curriculum changes. They are **random-policy** results,
not evidence about a trained checkpoint.

Run from the Podman dev image after `maturin develop --release` and with
`PYTHONPATH=/work/python`:

```bash
python -m warsim_rl.baseline --episodes 64 --opponent random
python -m warsim_rl.baseline --episodes 64 --opponent idle
```

Both runs used seed 12345, a 1v1 arena, 600 maximum ticks, and 16 parallel
environments. Sides alternated across environments. Contact means at least
one living opposing pair had line of sight during a tick. Shot and damage
totals include both teams; ending ammo is the sum for both units.

| opponent | wins / draws / losses | episodes with contact | shots / hits | kills | mean ending ammo | timeout matches |
|---|---:|---:|---:|---:|---:|---:|
| random | 0 / 64 / 0 | 64 / 64 | 3840 / 38 (0.99%) | 0 | 0 | 64 / 64 |
| idle | 0 / 64 / 0 | 59 / 64 | 1920 / 21 (1.09%) | 0 | 30 | 64 / 64 |

The random policy exhausted its 30 rounds per unit, yet almost never hit.
Visual contact occurred in most episodes, so inability to find an enemy was
not the main reason for these draws. A trained policy may behave differently;
these measurements only establish the random baseline for later comparisons.

The first curriculum stage is a closer 1v1 duel. With 64 episodes, seed 12345,
120 maximum ticks and the same random policies, the result was 16 wins,
35 draws and 13 losses. Contact occurred on every tick, 248 of 2,754 shots
hit (9.0%), and 29 matches ended in a team wipe. This stage gives training
a substantially denser combat signal; it is not evidence that the trained
policy improves. Reproduce with:

```bash
python -m warsim_rl.baseline --episodes 64 --opponent random --scenario duel --max-ticks 120
```

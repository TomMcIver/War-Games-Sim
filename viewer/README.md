# Replay viewer

A dependency-free, single-page browser viewer for War-Games-Sim replays.
No build step, no npm, no CDN: `index.html` + `viewer.js` + `replay-model.js`
+ `style.css`.

## Open it

- **Directly:** double-click `viewer/index.html`. The bundled demo
  (`sample_replay.js`) loads automatically; open your own replays with
  **Open replay** or drag-and-drop a `.json` file anywhere on the page.
- **Served** (needed for `?replay=` URLs, because browsers block `fetch()` on
  `file://`):
  - `bash infra/viewer.sh` (Podman, port 8000) then open
    `http://localhost:8000/viewer/`. Training output from `infra/train.sh` is
    served at `/runs/`, e.g. `http://localhost:8000/viewer/?replay=/runs/exp1/replays/ep10.json`.
  - or any static server: `python -m http.server -d viewer 8000`.

### URL parameters

| param | example | effect |
|---|---|---|
| `replay` | `?replay=/runs/x.json` | fetch and load that replay (served mode) |
| `t` | `&t=150` | jump to the first frame with tick >= 150 |
| `floor` | `&floor=1` | show floor 1 |
| `follow` | `&follow=2` | follow unit 2 |
| `play` | `&play=1` | start playing |

## Controls

| key / mouse | action |
|---|---|
| Space | play / pause |
| Left / Right | step one frame (Shift: 10) |
| Home / End | first / last frame |
| `+` / `-` | faster / slower (0.25x - 16x; 1x = real time from `tick_seconds`) |
| `[` `]`, PgDn / PgUp | floor down / up |
| `1`-`9` | follow unit 0-8 (press again to stop) |
| `F` / Esc | follow the hovered unit / stop following |
| `0`, double-click | fit map |
| wheel / drag | zoom / pan |
| click unit, or a row in *Forces* | follow it |
| `?` | shortcut help |

Hovering a unit shows id, team, hp, position, floor and facing; the bottom-left
line shows the tile under the cursor. The *Forces* panel shows alive counts and
HP per team/unit, *Combat* shows cumulative shots/hits/accuracy, and the floor
picker shows coloured dots for the living units on each floor.

## What it draws

| tile | look |
|---|---|
| Void (0) | nothing (background; on upper floors the floor below shows through, dimmed) |
| Floor (1) | dark checker |
| Wall (2) | light grey block with a shadow edge |
| DoorClosed (3) | brown slab across the doorway |
| DoorOpen (4) | leaf swung open + dotted swing arc |
| Window (5) | wall with a blue glass strip |
| StairsUp (6) / StairsDown (7) | treads with an amber up / down arrow |
| other ids | magenta (so unknown tiles are obvious) |

Doors and windows are oriented from their neighbouring walls. Units are
team-coloured circles (team 0 blue, team 1 red) with a facing line and an HP
bar; dead units are grey with a cross; units on other floors are dashed rings
with an up/down marker. Shots are tracers from the shooter's position in the
frame they were fired to the shot end point, fading over 6 frames, with a red
burst on a hit and a grey spark on a miss.

## Replay format (version 1)

Written by `sim_core::replay` (see `docs/INTERFACE.md`, "Replay JSON" and "Map JSON"):

```jsonc
{
  "version": 1,                      // required; other versions are rejected
  "config": {                        // free-form; these keys are used if present
    "tick_seconds": 0.1,             // playback speed (default 0.1)
    "map_kind": "arena",             // shown in the header
    "max_hp": 100,                   // full-health reference for HP bars
    "team_size": 2, "max_ticks": 600
  },
  "seed": 123,
  "map": {
    "width": 32, "height": 18, "floors": 2,
    "tiles": [1, 2, ...]             // W*H*F tile ids, index (f*H + y)*W + x
  },
  "frames": [                        // non-empty, strictly increasing tick
    {
      "tick": 0,
      "units": [                     // same unit ids in every frame
        {"id": 0, "team": 0, "x": 2.5, "y": 6.5, "floor": 0,
         "yaw": 0.0, "hp": 100.0, "alive": true}
      ],
      "shots": [                     // optional; shots fired during this tick
        {"shooter": 0, "x": 9.3, "y": 7.1, "floor": 0, "hit": 3}   // hit: unit id or null
      ]
    }
  ]
}
```

- Positions are continuous tile units: tile `(tx, ty)` covers `[tx, tx+1) x [ty, ty+1)`.
- `yaw` is radians; the facing vector is `(cos yaw, sin yaw)` in map coordinates
  (+x right, +y down on screen).
- `hp` is drawn relative to `config.max_hp`, else to the largest hp in the
  replay (if > 1), else to 1.
- Frames may be recorded every N ticks; playback speed adapts to the median tick step.
- The map is stored once, so doors opened/closed during the episode are drawn
  in their initial state (format v1 limitation).
- Invalid files are rejected with a list of the first problems found.

## Files

| file | role |
|---|---|
| `index.html`, `style.css` | layout and styling |
| `viewer.js` | canvas rendering, playback, input, panels |
| `replay-model.js` | pure logic (validation, interpolation, stats); also loaded by Node tests |
| `sample_replay.json` | bundled 2v2 demo on a 2-floor map (hand-choreographed, not sim output) |
| `sample_replay.js` | the same data as a script, so the demo works from `file://` |
| `tools/make_sample_replay.py` | regenerates both sample files |
| `tests/replay-model.test.js` | `node --test viewer/tests/*.test.js` (no npm packages) |

Regenerate the sample after changing the generator:

```bash
bash infra/run.sh python3 viewer/tools/make_sample_replay.py
```

The Rust test `replay::tests::bundled_viewer_sample_is_valid` and the Node tests
check that the sample matches the format and that both sample files agree.

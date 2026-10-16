"""Generate viewer/sample_replay.json (+ sample_replay.js) - a tiny 2v2 demo.

This is a hand-choreographed fixture for the viewer, NOT simulator output and
NOT a tactic: units follow fixed waypoints and fire on a fixed schedule so the
viewer has something to show before the engine exists. The JSON follows
docs/INTERFACE.md ("Map JSON", "Replay JSON") and is checked by the Rust test
`replay::tests::bundled_viewer_sample_is_valid`.

Run (no third-party deps):  bash infra/run.sh python3 viewer/tools/make_sample_replay.py
"""

import json
import math
from pathlib import Path

VOID, FLOOR, WALL, DOOR_CLOSED, DOOR_OPEN, WINDOW, STAIRS_UP, STAIRS_DOWN = range(8)
W, H, FLOORS = 32, 18, 2
LAST_TICK = 260
DAMAGE = 0.2
OUT_DIR = Path(__file__).resolve().parent.parent


def build_map():
    tiles = [[[VOID] * W for _ in range(H)] for _ in range(FLOORS)]
    g = tiles[0]
    for y in range(H):
        for x in range(W):
            border = x in (0, W - 1) or y in (0, H - 1)
            g[y][x] = WALL if border else FLOOR
    # Outdoor cover.
    for y in (5, 6, 7, 11, 12, 13):
        g[y][7] = WALL
    for y in (3, 4, 5, 12, 13, 14):
        g[y][26] = WALL
    # A two-storey building, x 13..21, y 4..13 (both floors share the footprint).
    for f in range(FLOORS):
        for y in range(4, 14):
            for x in range(13, 22):
                edge = x in (13, 21) or y in (4, 13)
                tiles[f][y][x] = WALL if edge else FLOOR
    g[9][13] = DOOR_OPEN
    g[8][21] = DOOR_OPEN
    g[13][17] = DOOR_CLOSED
    for x, y in ((15, 4), (19, 4), (13, 6), (13, 11)):
        g[y][x] = WINDOW
    for y in range(5, 9):
        g[y][17] = WALL  # interior partition with a gap at y 9..12
    g[11][19] = STAIRS_UP
    up = tiles[1]
    for x, y in ((13, 6), (13, 11), (21, 6), (17, 4)):
        up[y][x] = WINDOW
    up[11][19] = STAIRS_DOWN
    flat = [t for f in tiles for row in f for t in row]
    return {"width": W, "height": H, "floors": FLOORS, "tiles": flat}


# Waypoints: (tick, x, y, floor). Position is linearly interpolated between them.
PATHS = {
    0: [(0, 2.5, 6.5, 0), (30, 6.0, 6.0, 0), (200, 6.0, 6.0, 0), (225, 9.5, 8.0, 0)],
    1: [(0, 2.5, 11.5, 0), (30, 6.0, 12.5, 0), (70, 9.0, 15.5, 0), (140, 9.0, 15.5, 0),
        (175, 11.5, 15.0, 0)],
    2: [(0, 29.5, 8.5, 0), (35, 22.5, 8.5, 0), (45, 20.5, 8.5, 0), (65, 19.5, 11.5, 0),
        (66, 19.5, 11.5, 1), (90, 15.5, 6.5, 1), (110, 15.5, 6.5, 1), (130, 19.5, 11.5, 1),
        (131, 19.5, 11.5, 0), (150, 14.5, 10.5, 0)],
    3: [(0, 29.5, 10.5, 0), (30, 27.5, 13.5, 0), (60, 23.0, 15.5, 0), (100, 15.5, 15.5, 0),
        (115, 12.5, 15.5, 0)],
}
TEAMS = {0: 0, 1: 0, 2: 1, 3: 1}
# Fire schedules: (shooter, target, first tick, last tick, every n ticks, hit pattern).
SCHEDULE = [
    (3, 1, 96, 140, 7, "HMMH"),
    (1, 3, 98, 140, 6, "HMH"),
    (2, 1, 152, 230, 6, "MHH"),
    (0, 2, 158, 260, 7, "MHMH"),
    (2, 0, 186, 260, 8, "MMH"),
]


def waypoint_pos(path, tick):
    if tick <= path[0][0]:
        return path[0][1:]
    for (t0, x0, y0, f0), (t1, x1, y1, _f1) in zip(path, path[1:]):
        if t0 <= tick < t1:
            a = (tick - t0) / (t1 - t0)
            return (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, f0)
    return path[-1][1:]


def r3(v):
    return round(v, 3) + 0.0


def simulate():
    hp = {u: 1.0 for u in PATHS}
    frozen = {}  # unit -> (x, y, floor, yaw) at death
    yaw = {0: 0.0, 1: 0.0, 2: math.pi, 3: math.pi}
    fired = {i: 0 for i in range(len(SCHEDULE))}
    frames = []
    for tick in range(LAST_TICK + 1):
        pos = {u: frozen[u][:3] if u in frozen else waypoint_pos(PATHS[u], tick) for u in PATHS}
        prev = {u: frozen[u][:3] if u in frozen else waypoint_pos(PATHS[u], max(tick - 1, 0))
                for u in PATHS}
        aiming = {}
        shots = []
        for i, (shooter, target, start, end, every, pattern) in enumerate(SCHEDULE):
            if hp[shooter] <= 0 or hp[target] <= 0 or not start <= tick <= end:
                continue
            aiming[shooter] = target
            if (tick - start) % every:
                continue
            sx, sy, sf = pos[shooter]
            tx, ty, _ = pos[target]
            hit = pattern[fired[i] % len(pattern)] == "H"
            fired[i] += 1
            if hit:
                hp[target] = max(0.0, round(hp[target] - DAMAGE, 6))
                end_x, end_y = tx, ty
            else:  # a miss flies past the target, slightly off-line
                dx, dy = tx - sx, ty - sy
                n = math.hypot(dx, dy) or 1.0
                end_x = tx + dx / n * 2.0 - dy / n * 0.7
                end_y = ty + dy / n * 2.0 + dx / n * 0.7
            end_x = min(max(end_x, 0.5), W - 0.5)
            end_y = min(max(end_y, 0.5), H - 0.5)
            shots.append({"shooter": shooter, "x": r3(end_x), "y": r3(end_y), "floor": sf,
                          "hit": target if hit else None})
        units = []
        for u in sorted(PATHS):
            x, y, f = pos[u]
            if u in frozen:
                yaw_u = frozen[u][3]
            elif u in aiming:
                tx, ty, _ = pos[aiming[u]]
                yaw_u = math.atan2(ty - y, tx - x)
            elif (x, y) != prev[u][:2]:
                yaw_u = math.atan2(y - prev[u][1], x - prev[u][0])
            else:
                yaw_u = yaw[u]
            yaw[u] = yaw_u
            alive = hp[u] > 0
            if not alive and u not in frozen:
                frozen[u] = (x, y, f, yaw_u)
            units.append({"id": u, "team": TEAMS[u], "x": r3(x), "y": r3(y), "floor": f,
                          "yaw": r3(yaw_u), "hp": r3(hp[u]), "alive": alive})
        frames.append({"tick": tick, "units": units, "shots": shots})
        if any(all(hp[u] <= 0 for u in PATHS if TEAMS[u] == t) for t in (0, 1)):
            break  # episode ends when a team is wiped out
    return frames


def main():
    replay = {
        "version": 1,
        "config": {"team_size": 2, "max_ticks": 600, "tick_seconds": 0.1, "map_kind": "sample",
                   "note": "hand-made viewer demo, not simulator output"},
        "seed": 7,
        "map": build_map(),
        "frames": simulate(),
    }
    text = json.dumps(replay, separators=(",", ":"))
    (OUT_DIR / "sample_replay.json").write_text(text + "\n", encoding="utf-8")
    # Same data as a classic script so index.html works when opened via file://
    # (browsers block fetch() of local files).
    js = "// Generated by viewer/tools/make_sample_replay.py - do not edit.\n"
    js += "window.WARSIM_SAMPLE_REPLAY = " + text + ";\n"
    (OUT_DIR / "sample_replay.js").write_text(js, encoding="utf-8")
    print(f"wrote {len(replay['frames'])} frames, {len(text)} bytes")


if __name__ == "__main__":
    main()

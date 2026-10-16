# map module

Tile grid with floors, plus line of sight, bullet rays, movement collision,
reachability, Map JSON and seeded generators. The contract lives in
`docs/INTERFACE.md`.

## Coordinates
- Tile `(x, y, f)` covers `[x, x+1) x [y, y+1)` on floor `f`. A `Pos` is in
  tile `(floor(x), floor(y), floor)`. Tile centres are at `x + 0.5`.
- Flat index `(f * height + y) * width + x`. Reads outside the map return `Void`.
- Angles are radians, direction `(cos a, sin a)` in map x/y.

## Tile semantics
| tile | u8 | walk | sight | bullets | reachability (BFS) |
|------|----|------|-------|---------|--------------------|
| Void | 0 | no | blocks | blocks | no |
| Floor | 1 | yes | clear | clear | yes |
| Wall | 2 | no | blocks | blocks | no |
| DoorClosed | 3 | no | blocks | blocks | yes (a unit can open it) |
| DoorOpen | 4 | yes | clear | clear | yes |
| Window | 5 | no | clear | clear | no |
| StairsUp | 6 | yes | clear | clear | yes, plus a link to `(x, y, f+1)` |
| StairsDown | 7 | yes | clear | clear | yes, plus a link to `(x, y, f-1)` |

- `StairsUp` at `(x, y, f)` pairs with `StairsDown` at `(x, y, f+1)`
  (`stairs_destination`). Generated maps never contain unpaired stairs.
- `toggle_door(x, y, f)` switches open and closed and returns `false` for a
  tile that is not a door. The map does not track units, so the engine should
  not close a door that a unit overlaps.

## Queries
- `line_of_sight(a, b)`: exact Amanatides-Woo grid traversal with no
  allocation. Different floors always give `false`. The tiles that contain
  `a` and `b` never block, so a wall tile counts as visible when the line
  reaches it.
  Corner rule: a line that passes exactly through a grid vertex is blocked
  only if both tiles beside that vertex block. This seals walls that touch
  diagonally, lets a line graze a single wall corner, and keeps
  `los(a, b) == los(b, a)`.
- `raycast(from, angle, max_range) -> RayHit { distance, x, y, tile }`:
  stops at the first wall, closed door or Void tile. Windows do not stop it,
  and the shooter's own tile is ignored. With no hit it returns
  `distance = max_range` and `tile = None`.
- `circle_fits(x, y, f, r)` checks a circle against every non-walkable
  tile. Touching an edge exactly is allowed. `can_move(from, to, r)` sweeps
  the move in sub-steps of at most `r/2`, so it cannot tunnel through a wall,
  and it returns `false` for any floor change. `slide_move(from, dx, dy, r)`
  tries the full move, then each axis separately (larger component first),
  and moves up to the obstacle by bisection.
- `is_connected(a, b)`, `is_fully_connected()` and `validate()` do a BFS over
  the reachability column above.

## Generators
`generate(kind, seed, team_size)` uses only a seeded ChaCha8 RNG, so the
same inputs always give the same map. `team_size` is clamped to `1..=8`.
**An unknown kind falls back to `"arena"`.** Call `try_generate` to get
`MapError::UnknownKind` instead.

Every generated map is **point-symmetric**: a 180-degree rotation about the
centre maps the map onto itself. That makes it fair for self-play. Every map
also passes `validate()` (both spawn zones have at least 8 walkable tiles,
all mutually reachable, and all stairs are paired) and
`is_fully_connected()`. If an attempt fails these checks, the generator
retries on the same RNG stream. After 50 failed attempts it falls back to an
arena, which is valid by construction. This has not happened in tests.

| kind | size | content |
|------|------|---------|
| `arena` | `(23+2n) x (17+2n)`, 1 floor | Walled field with `3+n` to `6+n` cover blocks: walls, plus window "fences" that block movement but not sight. |
| `building` | 41 x 29, 2 floors | A 25x21 building in a walled yard. Each floor has a central spine wall with doors and BSP rooms (doors, interior windows). The ground floor has 2-3 exterior doors per half and windows. Two stair pairs. Yard cover. Floor 1 outside the building is Void. |
| `village` | 65 x 65, 2 floors | A street grid of 16 lots. About 80% hold a house of random size; about half of the houses that are at least 7x7 get a second floor with stairs. 18 to 26 cover blocks on streets and empty lots. |

Spawns: team 0 uses columns `1..=3` on the low-x edge, and team 1 uses the
rotation of those tiles. `spawn_points(team, n)` returns tile centres that
are distinct and walkable, spaced with Chebyshev distance of at least 2 where
possible, and starting near the centre row. It panics if `n` is larger than
the zone. Maps built by hand or with `from_json` derive a zone from the
walkable floor-0 tiles in the low-x third of the map.

Layout assumes a unit radius below 0.5: doorways and corridors are one tile
wide.

## JSON
`to_json()` / `from_json()` use exactly the contract shape
`{"width","height","floors","tiles":[u8]}`. Spawn zones are not serialized.

use super::{ascii, p, TestRng};
use crate::map::{generate, GameMap};
use crate::types::Pos;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

const ROOM: &[&str] = &[
    "#########", //
    "#.......#", //
    "#.......#", //
    "#.......#", //
    "#########",
];

#[test]
fn open_floor_has_sight() {
    let m = ascii(&[ROOM]);
    assert!(m.line_of_sight(p(1.5, 1.5), p(7.5, 3.5)));
    assert!(m.line_of_sight(p(1.2, 2.7), p(1.3, 2.9)), "same tile");
}

#[test]
fn walls_and_closed_doors_block_windows_and_open_doors_do_not() {
    for (mid, expect) in [
        ('#', false),
        ('+', false),
        (' ', false),
        ('=', true),
        ('/', true),
    ] {
        let row = format!("#...{mid}...#");
        let m = ascii(&[&["#########", &row, "#########"]]);
        assert_eq!(
            m.line_of_sight(p(1.5, 1.5), p(7.5, 1.5)),
            expect,
            "tile {mid:?}"
        );
    }
}

#[test]
fn cross_floor_sight_is_false() {
    let m = ascii(&[ROOM, ROOM]);
    let a = Pos {
        x: 2.5,
        y: 2.5,
        floor: 0,
    };
    let b = Pos {
        x: 2.5,
        y: 2.5,
        floor: 1,
    };
    assert!(!m.line_of_sight(a, b));
    assert!(m.line_of_sight(b, Pos { x: 5.5, ..b }));
}

#[test]
fn bad_inputs_have_no_sight() {
    let m = ascii(&[ROOM]);
    assert!(!m.line_of_sight(
        Pos {
            x: 2.5,
            y: 2.5,
            floor: 3
        },
        Pos {
            x: 3.5,
            y: 2.5,
            floor: 3
        }
    ));
    assert!(!m.line_of_sight(p(f32::NAN, 2.5), p(3.5, 2.5)));
    assert!(!m.line_of_sight(p(2.5, 2.5), p(f32::INFINITY, 2.5)));
}

#[test]
fn target_wall_tile_itself_is_visible_but_not_beyond() {
    let m = ascii(&[ROOM]);
    assert!(
        m.line_of_sight(p(4.5, 2.5), p(8.5, 2.5)),
        "wall tile is seen"
    );
    assert!(
        !m.line_of_sight(p(4.5, 2.5), p(10.5, 2.5)),
        "beyond the wall is not"
    );
}

#[test]
fn diagonal_wall_pair_blocks_through_shared_corner() {
    // Walls at (2,1) and (1,2) touch at vertex (2,2).
    let m = ascii(&[&[
        "#####", //
        "#.#.#", //
        "##..#", //
        "#...#", //
        "#####",
    ]]);
    assert!(!m.line_of_sight(p(1.5, 1.5), p(3.5, 3.5)));
    assert!(!m.line_of_sight(p(3.5, 3.5), p(1.5, 1.5)));
}

#[test]
fn grazing_a_single_wall_corner_does_not_block() {
    // Only (2,1) is a wall; the diagonal passes exactly through its corner.
    let m = ascii(&[&[
        "#####", //
        "#.#.#", //
        "#...#", //
        "#...#", //
        "#####",
    ]]);
    assert!(m.line_of_sight(p(1.5, 1.5), p(3.5, 3.5)));
    assert!(m.line_of_sight(p(3.5, 3.5), p(1.5, 1.5)));
    // Slightly off the corner into the wall: blocked.
    assert!(!m.line_of_sight(p(1.5, 1.4), p(3.5, 3.0)));
}

#[test]
fn axis_aligned_and_grid_line_segments() {
    let m = ascii(&[&[
        "#######", //
        "#.....#", //
        "#..#..#", //
        "#.....#", //
        "#######",
    ]]);
    assert!(!m.line_of_sight(p(1.5, 2.5), p(5.5, 2.5)));
    assert!(m.line_of_sight(p(1.5, 1.5), p(5.5, 1.5)));
    assert!(!m.line_of_sight(p(3.5, 1.5), p(3.5, 3.5)));
    // Exactly on the grid line y = 2.0: belongs to row 2, which has the wall.
    assert!(!m.line_of_sight(p(1.5, 2.0), p(5.5, 2.0)));
    // Exactly on y = 3.0 (row 3, all floor).
    assert!(m.line_of_sight(p(1.5, 3.0), p(5.5, 3.0)));
}

/// Exact reference: blocked iff the open segment crosses the open interior
/// of some blocking tile other than the endpoint tiles. `None` when the
/// segment only barely touches a blocking tile (numerically ambiguous).
fn reference_los(m: &GameMap, a: Pos, b: Pos) -> Option<bool> {
    let (ax, ay, bx, by) = (a.x as f64, a.y as f64, b.x as f64, b.y as f64);
    let (ta, tb) = (
        (ax.floor() as i32, ay.floor() as i32),
        (bx.floor() as i32, by.floor() as i32),
    );
    let mut ambiguous = false;
    for ty in ta.1.min(tb.1)..=ta.1.max(tb.1) {
        for tx in ta.0.min(tb.0)..=ta.0.max(tb.0) {
            if (tx, ty) == ta || (tx, ty) == tb || !m.blocks_sight(tx, ty, a.floor as i32) {
                continue;
            }
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for (s, d, c) in [(ax, bx - ax, tx as f64), (ay, by - ay, ty as f64)] {
                if d == 0.0 {
                    if s <= c || s >= c + 1.0 {
                        hi = -1.0;
                    }
                } else {
                    let (t0, t1) = ((c - s) / d, (c + 1.0 - s) / d);
                    lo = lo.max(t0.min(t1));
                    hi = hi.min(t0.max(t1));
                }
            }
            let len = hi - lo;
            if len.abs() < 1e-6 {
                ambiguous = true;
            } else if len > 0.0 {
                return Some(false);
            }
        }
    }
    if ambiguous {
        None
    } else {
        Some(true)
    }
}

#[test]
fn dda_matches_exact_reference_on_random_segments() {
    let m = generate("village", 7, 4);
    let (w, h) = (m.width as f32, m.height as f32);
    let mut rng = TestRng::new(1);
    let mut checked = 0;
    for _ in 0..20_000 {
        let a = p(rng.range(0.0, w), rng.range(0.0, h));
        let b = Pos {
            x: (a.x + rng.range(-12.0, 12.0)).clamp(0.0, w - 0.01),
            y: (a.y + rng.range(-12.0, 12.0)).clamp(0.0, h - 0.01),
            floor: 0,
        };
        if let Some(expect) = reference_los(&m, a, b) {
            assert_eq!(m.line_of_sight(a, b), expect, "{a:?} -> {b:?}");
            checked += 1;
        }
    }
    assert!(checked > 19_000);
}

#[test]
fn los_is_symmetric() {
    for kind in ["arena", "building", "village"] {
        let m = generate(kind, 3, 8);
        let (w, h) = (m.width as f32, m.height as f32);
        let mut rng = TestRng::new(2);
        for i in 0..20_000 {
            let floor = (i % m.floors) as u8;
            // Mix random points with tile centres and tile corners, which
            // exercise exact vertex crossings.
            let pick = |r: &mut TestRng| match i % 3 {
                0 => (r.range(0.0, w), r.range(0.0, h)),
                1 => (r.range(0.0, w).floor() + 0.5, r.range(0.0, h).floor() + 0.5),
                _ => (r.range(1.0, w).floor(), r.range(1.0, h).floor()),
            };
            let (ax, ay) = pick(&mut rng);
            let (bx, by) = pick(&mut rng);
            let a = Pos {
                x: ax,
                y: ay,
                floor,
            };
            let b = Pos {
                x: bx,
                y: by,
                floor,
            };
            assert_eq!(
                m.line_of_sight(a, b),
                m.line_of_sight(b, a),
                "{kind}: {a:?} {b:?}"
            );
        }
    }
}

#[test]
fn raycast_stops_at_walls_and_closed_doors_but_not_windows() {
    let m = ascii(&[&[
        "############", //
        "#..=..+...##", //
        "#..=../...##", //
        "############",
    ]]);
    // Window at x=3 is passed, closed door at x=6 stops the ray.
    let hit = m.raycast(p(1.5, 1.5), 0.0, 50.0);
    assert!((hit.distance - 4.5).abs() < 1e-4, "{hit:?}");
    assert_eq!(hit.tile, Some((6, 1)));
    assert!((hit.x - 6.0).abs() < 1e-4 && (hit.y - 1.5).abs() < 1e-4);
    // Open door passes; wall at x=10 stops.
    let hit = m.raycast(p(1.5, 2.5), 0.0, 50.0);
    assert!((hit.distance - 8.5).abs() < 1e-4, "{hit:?}");
    assert_eq!(hit.tile, Some((10, 2)));
    // Facing -x (angle pi) hits the outer wall at x=1.
    let hit = m.raycast(p(1.5, 2.5), PI, 50.0);
    assert!((hit.distance - 0.5).abs() < 1e-4, "{hit:?}");
    assert_eq!(hit.tile, Some((0, 2)));
    // Facing +y hits the bottom wall at y=3.
    let hit = m.raycast(p(4.5, 1.25), FRAC_PI_2, 50.0);
    assert!((hit.distance - 1.75).abs() < 1e-4, "{hit:?}");
    assert_eq!(hit.tile, Some((4, 3)));
}

#[test]
fn raycast_respects_max_range() {
    let m = ascii(&[ROOM]);
    let hit = m.raycast(p(1.5, 2.5), 0.0, 3.0);
    assert!(!hit.blocked());
    assert!((hit.distance - 3.0).abs() < 1e-5);
    assert!((hit.x - 4.5).abs() < 1e-5 && (hit.y - 2.5).abs() < 1e-5);
    // Range ending exactly inside the wall tile still reports the wall.
    let hit = m.raycast(p(1.5, 2.5), 0.0, 7.2);
    assert_eq!(hit.tile, Some((8, 2)));
    assert!((hit.distance - 6.5).abs() < 1e-4);
    // Degenerate inputs.
    assert_eq!(m.raycast(p(1.5, 2.5), 0.0, 0.0).distance, 0.0);
    assert_eq!(m.raycast(p(1.5, 2.5), 0.0, -1.0).distance, 0.0);
    assert_eq!(m.raycast(p(f32::NAN, 2.5), 0.0, 5.0).distance, 0.0);
}

#[test]
fn raycast_diagonal_distance() {
    let m = ascii(&[ROOM]);
    // From (1.5, 1.5) at 45 degrees: enters the bottom wall row y=4 at
    // t = 2.5 * sqrt(2).
    let hit = m.raycast(p(1.5, 1.5), FRAC_PI_4, 50.0);
    assert!((hit.distance - 2.5 * 2f32.sqrt()).abs() < 1e-3, "{hit:?}");
    assert_eq!(hit.tile.map(|t| t.1), Some(4));
}

#[test]
fn raycast_agrees_with_line_of_sight() {
    let m = generate("building", 11, 4);
    let (w, h) = (m.width as f32, m.height as f32);
    let mut rng = TestRng::new(3);
    for _ in 0..5_000 {
        let a = p(rng.range(0.0, w), rng.range(0.0, h));
        let b = p(rng.range(0.0, w), rng.range(0.0, h));
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-3 || !m.is_walkable(a.x as i32, a.y as i32, 0) {
            continue;
        }
        let hit = m.raycast(a, dy.atan2(dx), dist);
        if m.line_of_sight(a, b) {
            // A clear line means the ray can only stop in b's own tile.
            assert!(
                !hit.blocked() || hit.distance > dist - 1.5,
                "{a:?} {b:?} {hit:?}"
            );
        } else {
            assert!(
                hit.blocked() && hit.distance < dist + 1e-3,
                "{a:?} {b:?} {hit:?}"
            );
        }
    }
}

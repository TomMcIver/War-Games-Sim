use super::{ascii, p, TestRng};
use crate::map::generate;
use crate::types::Pos;

const R: f32 = 0.3;

const CORRIDOR: &[&str] = &[
    "##########", //
    "#....=...#", //
    "#....+...#", //
    "#..../...#", //
    "#........#", //
    "##########",
];

#[test]
fn free_move_on_open_floor() {
    let m = ascii(&[CORRIDOR]);
    assert!(m.can_move(p(1.5, 4.5), p(8.5, 4.5), R));
    assert!(m.can_move(p(1.5, 1.5), p(1.5, 4.5), R));
}

#[test]
fn walls_windows_and_closed_doors_block_movement() {
    let m = ascii(&[CORRIDOR]);
    assert!(!m.can_move(p(4.5, 1.5), p(6.5, 1.5), R), "window");
    assert!(!m.can_move(p(4.5, 2.5), p(6.5, 2.5), R), "closed door");
    assert!(m.can_move(p(4.5, 3.5), p(6.5, 3.5), R), "open door");
    assert!(!m.can_move(p(1.5, 4.5), p(1.5, 5.5), R), "outer wall");
    assert!(!m.can_move(p(1.5, 4.5), p(-3.0, 4.5), R), "outside the map");
}

#[test]
fn circle_overlap_uses_radius() {
    let m = ascii(&[CORRIDOR]);
    // Wall tile x=0 spans [0, 1); a circle at x=1.25 with r=0.25 just touches.
    assert!(m.circle_fits(1.25, 4.5, 0, 0.25));
    assert!(!m.circle_fits(1.2, 4.5, 0, 0.25));
    // Pillar at (2,2): near its corner (2,2) the distance is Euclidean.
    let pillar = ascii(&[&["#####", "#...#", "#.#.#", "#...#", "#####"]]);
    assert!(pillar.circle_fits(1.75, 1.75, 0, 0.3), "0.354 from the corner");
    assert!(!pillar.circle_fits(1.8, 1.8, 0, 0.3), "0.283 from the corner");
    // Centre in a blocked tile never fits, even with zero radius.
    assert!(!m.circle_fits(0.5, 4.5, 0, 0.0));
    assert!(m.circle_fits(1.01, 4.5, 0, 0.0));
    assert!(!m.circle_fits(f32::NAN, 4.5, 0, 0.1));
}

#[test]
fn large_steps_cannot_tunnel_through_thin_walls() {
    let m = ascii(&[CORRIDOR]);
    assert!(!m.can_move(p(4.5, 1.5), p(7.5, 1.5), R));
    assert!(!m.can_move(p(4.5, 2.5), p(8.5, 2.5), 0.01));
}

#[test]
fn floor_changes_are_not_moves() {
    let m = ascii(&[CORRIDOR, CORRIDOR]);
    let a = p(2.5, 4.5);
    assert!(!m.can_move(a, Pos { floor: 1, ..a }, R));
}

#[test]
fn slide_move_slides_along_a_wall() {
    let m = ascii(&[CORRIDOR]);
    // Moving down-left into the left wall: x stops at 1 + R, y moves fully.
    let out = m.slide_move(p(1.6, 2.0), -1.0, 0.5, R);
    assert!((out.x - 1.3).abs() < 0.01, "{out:?}");
    assert!((out.y - 2.5).abs() < 1e-5, "{out:?}");
    assert!(m.circle_fits(out.x, out.y, 0, R));
    // Free move is unchanged.
    let out = m.slide_move(p(2.5, 4.5), 0.2, -0.1, R);
    assert!((out.x - 2.7).abs() < 1e-6 && (out.y - 4.4).abs() < 1e-6);
    // Head-on into a wall: stays (or creeps up to it) but never enters.
    let out = m.slide_move(p(1.5, 4.5), 0.0, 1.0, R);
    assert!((out.y - (5.0 - R)).abs() < 0.01 && (out.x - 1.5).abs() < 1e-6, "{out:?}");
    // Non-finite delta is ignored.
    assert_eq!(m.slide_move(p(1.5, 4.5), f32::NAN, 0.0, R), p(1.5, 4.5));
}

#[test]
fn random_walks_never_end_inside_obstacles() {
    for kind in ["arena", "building", "village"] {
        let m = generate(kind, 5, 8);
        let mut rng = TestRng::new(9);
        for start in m.spawn_points(0, 8).into_iter().chain(m.spawn_points(1, 8)) {
            let mut pos = start;
            for _ in 0..400 {
                let (dx, dy) = (rng.range(-0.4, 0.4), rng.range(-0.4, 0.4));
                pos = m.slide_move(pos, dx, dy, 0.35);
                assert!(m.circle_fits(pos.x, pos.y, pos.floor as i32, 0.35), "{kind} {pos:?}");
            }
        }
    }
}

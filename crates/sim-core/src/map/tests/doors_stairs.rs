use super::{ascii, p};
use crate::map::{generate, Tile};
use crate::types::Pos;

#[test]
fn toggle_door_flips_state_and_changes_sight() {
    let mut m = ascii(&[&["#######", "#..+..#", "#######"]]);
    let (a, b) = (p(1.5, 1.5), p(5.5, 1.5));
    assert!(!m.line_of_sight(a, b));
    assert!(m.toggle_door(3, 1, 0));
    assert_eq!(m.tile(3, 1, 0), Tile::DoorOpen);
    assert!(m.line_of_sight(a, b));
    assert!(m.is_walkable(3, 1, 0));
    assert!(m.toggle_door(3, 1, 0));
    assert_eq!(m.tile(3, 1, 0), Tile::DoorClosed);
    assert!(!m.line_of_sight(a, b));
}

#[test]
fn toggle_door_rejects_non_doors() {
    let mut m = ascii(&[&["#######", "#..+..#", "#######"]]);
    let before = m.clone();
    assert!(!m.toggle_door(1, 1, 0), "floor");
    assert!(!m.toggle_door(0, 0, 0), "wall");
    assert!(!m.toggle_door(-1, 1, 0), "out of bounds");
    assert!(!m.toggle_door(3, 1, 4), "bad floor");
    assert_eq!(m, before);
}

#[test]
fn stairs_pair_across_floors() {
    let m = ascii(&[
        &["#####", "#U.U#", "#####"], //
        &["#####", "#D.##", "#####"],
    ]);
    assert_eq!(m.stairs_destination(1, 1, 0), Some((1, 1, 1)));
    assert_eq!(m.stairs_destination(1, 1, 1), Some((1, 1, 0)));
    assert_eq!(m.stairs_destination(3, 1, 0), None, "no StairsDown above");
    assert_eq!(m.stairs_destination(2, 1, 0), None, "not stairs");
    assert_eq!(m.stairs_destination(1, 1, 5), None, "out of bounds");
    assert!(m.validate().is_err(), "unpaired stairs fail validation");
}

#[test]
fn is_connected_follows_doors_and_stairs() {
    let m = ascii(&[
        &["#########", "#U.#..+.#", "#..#..#.#", "#########"],
        &["#########", "#D.....##", "####.####", "#########"],
    ]);
    let left = p(2.5, 2.5);
    // Floor 1 has only the one staircase, so the middle floor-0 room is
    // not reachable from the left room.
    assert!(!m.is_connected(left, p(4.5, 1.5)));
    assert!(m.is_connected(
        left,
        Pos {
            x: 4.5,
            y: 2.5,
            floor: 1
        }
    ));
    // Closed door counts as passable: (5,1) -> (7,1) through (6,1).
    assert!(m.is_connected(p(4.5, 1.5), p(7.5, 2.5)));
    // Non-passable endpoints are never connected.
    assert!(!m.is_connected(left, p(3.5, 1.5)));
    assert!(!m.is_connected(p(f32::NAN, 1.0), left));
    assert!(!m.is_fully_connected());
}

#[test]
fn generated_stairs_are_paired_and_reversible() {
    for kind in ["building", "village"] {
        for seed in 0..10 {
            let m = generate(kind, seed, 4);
            let mut ups = 0;
            for f in 0..m.floors as i32 {
                for y in 0..m.height as i32 {
                    for x in 0..m.width as i32 {
                        let t = m.tile(x, y, f);
                        if !t.is_stairs() {
                            continue;
                        }
                        ups += usize::from(t == Tile::StairsUp);
                        let (dx, dy, df) = m.stairs_destination(x, y, f).expect("paired");
                        assert_eq!((dx, dy), (x, y));
                        assert_eq!(m.stairs_destination(dx, dy, df), Some((x, y, f)));
                    }
                }
            }
            if kind == "building" {
                assert_eq!(ups, 2, "building has two stair pairs");
            }
        }
    }
}

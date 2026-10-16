use crate::map::{generate, try_generate, GameMap, MapError, MapKind, Tile};
use crate::types::MAX_TEAM_SIZE;

const KINDS: [&str; 3] = ["arena", "building", "village"];
const SEEDS: u64 = 40;

fn count(m: &GameMap, pred: impl Fn(Tile) -> bool) -> usize {
    m.tiles().iter().filter(|&&t| pred(t)).count()
}

#[test]
fn every_generator_is_valid_and_connected_for_many_seeds() {
    for kind in KINDS {
        for seed in 0..SEEDS {
            let ts = 1 + (seed as usize % MAX_TEAM_SIZE);
            let m = generate(kind, seed, ts);
            assert_eq!(m.validate(), Ok(()), "{kind} seed {seed}");
            assert!(m.is_fully_connected(), "{kind} seed {seed}");
        }
    }
}

#[test]
fn generators_never_fall_back_to_arena() {
    for seed in 0..SEEDS {
        let b = generate("building", seed, 8);
        assert_eq!(
            (b.width, b.height, b.floors),
            (41, 29, 2),
            "building seed {seed}"
        );
        let v = generate("village", seed, 8);
        assert_eq!(
            (v.width, v.height, v.floors),
            (65, 65, 2),
            "village seed {seed}"
        );
    }
}

#[test]
fn same_seed_gives_identical_map_and_seeds_differ() {
    for kind in KINDS {
        let a = generate(kind, 1234, 4);
        assert_eq!(a, generate(kind, 1234, 4), "{kind}");
        assert_eq!(a.to_json(), generate(kind, 1234, 4).to_json());
        let distinct = (0..8).filter(|&s| generate(kind, s, 4) != a).count();
        assert_eq!(distinct, 8, "{kind}: other seeds should differ");
    }
}

#[test]
fn generated_maps_are_point_symmetric() {
    for kind in KINDS {
        for seed in 0..10 {
            let m = generate(kind, seed, 4);
            for f in 0..m.floors as i32 {
                for y in 0..m.height as i32 {
                    for x in 0..m.width as i32 {
                        let (rx, ry, _) = m.rotate(x, y, f);
                        assert_eq!(
                            m.tile(x, y, f),
                            m.tile(rx, ry, f),
                            "{kind} {seed} {x},{y},{f}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn spawn_points_are_valid_distinct_opposite_and_reachable() {
    for kind in KINDS {
        for seed in 0..15 {
            let m = generate(kind, seed, 8);
            let mid = m.width as f32 / 2.0;
            for n in 1..=MAX_TEAM_SIZE {
                let t0 = m.spawn_points(0, n);
                let t1 = m.spawn_points(1, n);
                assert_eq!((t0.len(), t1.len()), (n, n));
                let all: Vec<_> = t0.iter().chain(&t1).collect();
                for (i, a) in all.iter().enumerate() {
                    assert!(
                        m.circle_fits(a.x, a.y, a.floor as i32, 0.45),
                        "{kind} {seed} {a:?}"
                    );
                    for b in &all[i + 1..] {
                        assert!(a != b, "{kind} {seed}: duplicate spawn {a:?}");
                    }
                    assert!(
                        m.is_connected(**a, *all[0]),
                        "{kind} {seed}: {a:?} unreachable"
                    );
                }
                assert!(t0.iter().all(|q| q.x < mid) && t1.iter().all(|q| q.x > mid));
            }
        }
    }
}

#[test]
fn spawn_points_are_spread_out() {
    let m = generate("arena", 0, 8);
    let pts = m.spawn_points(0, 8);
    for (i, a) in pts.iter().enumerate() {
        for b in &pts[i + 1..] {
            assert!(
                (a.x - b.x).abs().max((a.y - b.y).abs()) >= 2.0,
                "{a:?} {b:?}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "spawn_points")]
fn spawn_points_panics_when_zone_too_small() {
    let m = generate("arena", 0, 1);
    let _ = m.spawn_points(0, 10_000);
}

#[test]
fn hand_built_maps_still_have_spawns() {
    let m = GameMap::arena(20, 12);
    assert_eq!(m.validate(), Ok(()));
    let t0 = m.spawn_points(0, 8);
    let t1 = m.spawn_points(1, 8);
    assert!(t0.iter().all(|q| q.x < 4.0) && t1.iter().all(|q| q.x > 16.0));
    let open = GameMap::filled(30, 10, 1, Tile::Floor);
    assert_eq!(open.spawn_points(1, 3).len(), 3);
}

#[test]
fn arena_scales_with_team_size_and_clamps() {
    assert_eq!(generate("arena", 0, 1).width, 25);
    assert_eq!(generate("arena", 0, 8).width, 39);
    assert_eq!(generate("arena", 0, 0), generate("arena", 0, 1));
    assert_eq!(generate("arena", 0, 99), generate("arena", 0, 8));
    let m = generate("arena", 3, 4);
    assert!(
        count(&m, |t| t == Tile::Wall) > 2 * (m.width + m.height),
        "has cover"
    );
}

#[test]
fn building_has_rooms_doors_windows_and_two_floors() {
    for seed in 0..10 {
        let m = generate("building", seed, 4);
        assert!(count(&m, Tile::is_door) >= 6, "seed {seed}");
        assert!(count(&m, |t| t == Tile::Window) >= 2, "seed {seed}");
        let upper = (0..m.height as i32)
            .flat_map(|y| (0..m.width as i32).map(move |x| (x, y)))
            .filter(|&(x, y)| m.is_walkable(x, y, 1))
            .count();
        assert!(
            upper > 100,
            "seed {seed}: floor 1 has {upper} walkable tiles"
        );
        // Upper floor outside the building is Void.
        assert_eq!(m.tile(2, 2, 1), Tile::Void);
    }
}

#[test]
fn village_has_several_houses() {
    let mut two_storey = 0;
    for seed in 0..10 {
        let m = generate("village", seed, 4);
        let exterior_doors = (0..m.height as i32)
            .flat_map(|y| (0..m.width as i32).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                m.tile(x, y, 0).is_door()
                    && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        .iter()
                        .any(|&(dx, dy)| m.tile(x + dx, y + dy, 0) == Tile::Floor)
            })
            .count();
        assert!(exterior_doors >= 6, "seed {seed}: {exterior_doors}");
        two_storey += count(&m, |t| t == Tile::StairsUp);
    }
    assert!(two_storey > 0, "some village houses have an upper floor");
}

#[test]
fn unknown_kind_errors_or_falls_back_to_arena() {
    assert_eq!(
        try_generate("castle", 1, 2).unwrap_err(),
        MapError::UnknownKind("castle".into())
    );
    assert!(MapError::UnknownKind("castle".into())
        .to_string()
        .contains("arena"));
    assert_eq!(generate("castle", 1, 2), generate("arena", 1, 2));
    assert_eq!(
        try_generate(" Village ", 1, 2).unwrap(),
        generate("village", 1, 2)
    );
    assert_eq!("BUILDING".parse::<MapKind>(), Ok(MapKind::Building));
    assert_eq!(
        GameMap::generate("building", 4, 2),
        generate("building", 4, 2)
    );
    for name in MapKind::NAMES {
        assert_eq!(name.parse::<MapKind>().unwrap().name(), name);
    }
}

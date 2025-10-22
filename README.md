# SimCore Development

## Build Requirements

- CMake 3.12+
- C++17 compiler
- EnTT (fetched automatically via CMake FetchContent)

## Build Instructions

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/apps/sim-cli/sim-cli
```

On Windows:
```bash
.\build\apps\sim-cli\Debug\sim-cli.exe
```

## Project Structure

- `sim-core/` - Core simulation library (static)
  - `engine.h/cpp` - Main simulation engine
  - `tick_loop.h/cpp` - Fixed-timestep loop helper
  - `world.h/cpp` - ECS world wrapper (entt::registry)
  - `systems/` - ECS systems
    - `kinematics.h/cpp` - Position integration system
- `shared/` - Shared types and utilities
  - `types.h` - Basic type aliases (EntityId, Team, Vec2/Vec3)
  - `components.h` - ECS components (Transform, Kinematics, TeamTag)
  - `serialization.h/cpp` - Binary serialization utilities
- `apps/sim-cli/` - Command-line application
  - `main.cpp` - Demo that runs 100 ticks

## Build Configuration

- Default build type: `RelWithDebInfo`
- Compiler flags: `-Wall -Wextra -Werror -pedantic` (or `/W4 /WX` on MSVC)
- C++ standard: C++17 (required)

## Key Features

- Deterministic simulation with seeded RNG
- EnTT entity-component-system architecture
- Zero I/O beyond stdout
- Fixed-timestep tick loop (double precision internally)
- Command queue system with processing counter
- Binary state snapshot serialization

## Serialization Format

All binary data is **little-endian**.

### State Snapshot Format (Version 1)

```
Header:
  u32   Magic number (0x53494D53 = "SIMS")
  u16   Version (1)
  u64   Current tick

Entity Data:
  u32   Entity count

For each entity:
  u64   Entity ID

  Transform component:
    f64   Position X
    f64   Position Y
    f64   Position Z
    f32   Yaw rotation
    u32   Floor level

  TeamTag component (optional):
    u8    Has TeamTag flag (0 = no, 1 = yes)
    u8    Team ID (0 = Red, 1 = Blue) [only if flag = 1]
```

### Version History

- **Version 1** (current): Initial snapshot format with Transform and TeamTag components

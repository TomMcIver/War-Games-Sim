# SimCore Development

## Build Requirements

- CMake 3.12+
- C++17 compiler
- No external dependencies

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
- `shared/` - Shared types and utilities
  - `types.h` - Basic type aliases (EntityId, Team, Vec2/Vec3)
- `apps/sim-cli/` - Command-line application
  - `main.cpp` - Demo that runs 100 ticks

## Build Configuration

- Default build type: `RelWithDebInfo`
- Compiler flags: `-Wall -Wextra -Werror -pedantic` (or `/W4 /WX` on MSVC)
- C++ standard: C++17 (required)

## Key Features

- Deterministic simulation with seeded RNG
- Zero I/O beyond stdout
- Fixed-timestep tick loop
- Command queue system
- State snapshot capability

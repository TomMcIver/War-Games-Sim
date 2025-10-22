#include "sim-core/engine.h"
#include "sim-core/tick_loop.h"
#include "sim-core/world.h"
#include "shared/components.h"
#include <iostream>
#include <vector>

int main() {
  SimConfig config{0.05, 12345}; // 0.05s per tick, seed 12345
  Engine engine(config);

  // spawn Blue entity at (0,0,0) with velocity (1,0,0) and maxSpeed respected
  World& world = engine.getWorld();
  EntityId entity = world.spawn();

  auto& registry = world.registry();
  registry.emplace<Transform>(static_cast<entt::entity>(entity), Transform{{0.0, 0.0, 0.0}, 0.0f, 0});
  registry.emplace<Kinematics>(static_cast<entt::entity>(entity), Kinematics{{1.0, 0.0, 0.0}, 10.0f});
  registry.emplace<TeamTag>(static_cast<entt::entity>(entity), TeamTag{TeamTag::Team::Blue});

  // Run 100 ticks, enqueue fake command every 20 ticks
  for (int i = 1; i <= 100; ++i) {
    // enqueue fake command every 20 ticks
    if (i % 20 == 0) {
      std::vector<uint8_t> command = {0x01, 0x00};
      engine.enqueueUserCommand(command);
    }

    TickLoop::runFixed(engine, config.tickSeconds);
  }

  // final position
  const auto& transform = registry.get<Transform>(static_cast<entt::entity>(entity));
  std::cout << "After 100 ticks at 0.05s step:\n";
  std::cout << "Final position: (" << transform.pos.x << ", "
            << transform.pos.y << ", " << transform.pos.z << ")\n";
  std::cout << "Expected x ≈ 5.0, actual x = " << transform.pos.x << "\n\n";

  // check command processing
  std::cout << "Commands processed: " << engine.getProcessedCommandCount() << "\n";
  std::cout << "Expected: 5 commands (at ticks 20, 40, 60, 80, 100)\n";

  return 0;
}

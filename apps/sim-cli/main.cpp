#include "sim-core/engine.h"
#include "sim-core/tick_loop.h"
#include "sim-core/world.h"
#include "shared/components.h"
#include "shared/orders.h"
#include "shared/serialization.h"
#include <iostream>
#include <vector>

int main() {
  SimConfig config{0.05, 12345}; // 0.05s per tick, seed 12345
  Engine engine(config);

  World& world = engine.getWorld();
  auto& registry = world.registry();

  // creat 3 entities
  std::vector<EntityId> entities;
  for (int i = 0; i < 3; ++i) {
    EntityId entity = world.spawn();
    entities.push_back(entity);

    registry.emplace<Transform>(static_cast<entt::entity>(entity),
      Transform{{static_cast<double>(i * 2), 0.0, 0.0}, 0.0f, 0});
    registry.emplace<TeamTag>(static_cast<entt::entity>(entity),
      TeamTag{TeamTag::Team::Blue});
  }

  // push move order to (10,0,0) for each entiy
  std::cout << "=== Assigning Move orders to 3 entities ===\n";
  for (size_t i = 0; i < entities.size(); ++i) {
    Order order;
    order.type = OrderType::Move;
    order.target = Vec3{10.0, 0.0, 0.0};
    order.param = 0.0f;
    order.entityId = static_cast<uint32_t>(entities[i]);

    // serialze order into command buffer
    ByteBuffer buffer;
    OrderSerialization::writeOrder(buffer, order);

    std::vector<uint8_t> command(buffer.size());
    std::memcpy(command.data(), buffer.data(), buffer.size());

    engine.enqueueUserCommand(command);
  }

  std::cout << "\n=== Running simulation ===\n";
  // run 5 ticks to see orders being consumd
  for (int tick = 1; tick <= 5; ++tick) {
    std::cout << "\n--- Tick " << tick << " ---\n";
    TickLoop::runFixed(engine, config.tickSeconds);
  }

  std::cout << "\n=== Summary ===\n";
  std::cout << "Commands processed: " << engine.getProcessedCommandCount() << "\n";

  // check remaining orders
  auto ordersView = registry.view<Orders>();
  for (auto entity : ordersView) {
    const auto& orders = ordersView.get<Orders>(entity);
    std::cout << "Entity " << static_cast<uint64_t>(entity)
              << " has " << orders.queue.size() << " orders remaining\n";
  }

  return 0;
}

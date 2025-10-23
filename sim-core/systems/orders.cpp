#include "orders.h"
#include "sim-core/world.h"
#include "shared/components.h"
#include "shared/orders.h"
#include "shared/serialization.h"
#include <entt/entt.hpp>
#include <iostream>

void OrderSystem::processCommands(World& world, const std::vector<std::vector<uint8_t>>& commands) {
  auto& registry = world.registry();

  // parse each comand buffer and asign orders to entities
  for (const auto& commandBytes : commands) {
    if (commandBytes.empty()) continue;

    // for now, assume each comand is a serialized order
    // in a real systm, you'd have command type dispatch here
    // order size: 1 (type) + 24 (3x f64) + 4 (f32) + 4 (u32) = 33 bytes
    if (commandBytes.size() >= 33) {
      size_t offset = 0;
      Order order = OrderSerialization::readOrder(commandBytes.data(), offset);

      // find entity and add order to its qeue
      auto entity = static_cast<entt::entity>(order.entityId);
      if (registry.valid(entity)) {
        // ensure entiy has orders component
        if (!registry.all_of<Orders>(entity)) {
          registry.emplace<Orders>(entity);
        }

        auto& orders = registry.get<Orders>(entity);
        orders.queue.push_back(order);
      }
    }
  }
}

void OrderSystem::consumeOrders(World& world) {
  auto& registry = world.registry();
  auto view = registry.view<Orders>();

  // consume one order per entiy (just log for now)
  for (auto entity : view) {
    auto& orders = view.get<Orders>(entity);

    if (!orders.queue.empty()) {
      const auto& order = orders.queue.front();

      // log order consumtion
      std::cout << "Entity " << static_cast<uint64_t>(entity)
                << " consuming order (type=" << static_cast<int>(order.type)
                << ", target=[" << order.target.x << "," << order.target.y << "," << order.target.z << "]"
                << "), remaining orders: " << (orders.queue.size() - 1) << "\n";

      // TODO: actualy execute the order (pathfinding, etc.)

      // remove consumed order
      orders.queue.pop_front();
    }
  }
}

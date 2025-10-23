#pragma once

#include <vector>
#include <cstdint>

class World;
struct Order;

class OrderSystem {
public:
  // proces raw command buffers and asign orders to entities
  static void processCommands(World& world, const std::vector<std::vector<uint8_t>>& commands);

  // consume one order per entiy (stub for now - just logs)
  static void consumeOrders(World& world);
};

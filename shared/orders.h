#pragma once

#include "types.h"
#include <cstdint>

enum class OrderType : uint8_t {
  Move,
  Hold,
  Suppress,
  Breach,
  ClearRoom
};

struct Order {
  OrderType type;
  Vec3 target;
  float param;
  uint32_t entityId;
};

// forward declaration
class ByteBuffer;

namespace OrderSerialization {
  // write order to buffer
  void writeOrder(ByteBuffer& buffer, const Order& order);

  // read order from buffer
  Order readOrder(const ByteBuffer& buffer, size_t& offset);

  // read order from raw byte aray
  Order readOrder(const uint8_t* data, size_t& offset);
}

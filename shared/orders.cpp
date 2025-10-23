#include "orders.h"
#include "serialization.h"

namespace OrderSerialization {

void writeOrder(ByteBuffer& buffer, const Order& order) {
  buffer.write_u8(static_cast<uint8_t>(order.type));
  buffer.write_f64(order.target.x);
  buffer.write_f64(order.target.y);
  buffer.write_f64(order.target.z);
  buffer.write_f32(order.param);
  buffer.write_u32(order.entityId);
}

Order readOrder(const ByteBuffer& buffer, size_t& offset) {
  Order order;
  order.type = static_cast<OrderType>(buffer.read_u8(offset));
  order.target.x = buffer.read_f64(offset);
  order.target.y = buffer.read_f64(offset);
  order.target.z = buffer.read_f64(offset);
  order.param = buffer.read_f32(offset);
  order.entityId = buffer.read_u32(offset);
  return order;
}

Order readOrder(const uint8_t* data, size_t& offset) {
  Order order;
  order.type = static_cast<OrderType>(ByteBuffer::read_u8(data, offset));
  order.target.x = ByteBuffer::read_f64(data, offset);
  order.target.y = ByteBuffer::read_f64(data, offset);
  order.target.z = ByteBuffer::read_f64(data, offset);
  order.param = ByteBuffer::read_f32(data, offset);
  order.entityId = ByteBuffer::read_u32(data, offset);
  return order;
}

} // namespace OrderSerialization

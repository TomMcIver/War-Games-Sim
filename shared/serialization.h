#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

// Grow byte buffer for serialization
class ByteBuffer {
public:
  ByteBuffer() = default;

  void clear() { data_.clear(); }
  size_t size() const { return data_.size(); }
  const uint8_t* data() const { return data_.data(); }

  // Write primitives 
  void write_u8(uint8_t value);
  void write_u16(uint16_t value);
  void write_u32(uint32_t value);
  void write_u64(uint64_t value);
  void write_f32(float value);
  void write_f64(double value);

  // Read  (
  uint8_t read_u8(size_t& offset) const;
  uint16_t read_u16(size_t& offset) const;
  uint32_t read_u32(size_t& offset) const;
  uint64_t read_u64(size_t& offset) const;
  float read_f32(size_t& offset) const;
  double read_f64(size_t& offset) const;

private:
  std::vector<uint8_t> data_;
};

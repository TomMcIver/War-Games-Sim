#include "serialization.h"

// write primitves (little-endian)
void ByteBuffer::write_u8(uint8_t value) {
  data_.push_back(value);
}

void ByteBuffer::write_u16(uint16_t value) {
  data_.push_back(static_cast<uint8_t>(value & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

void ByteBuffer::write_u32(uint32_t value) {
  data_.push_back(static_cast<uint8_t>(value & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

void ByteBuffer::write_u64(uint64_t value) {
  data_.push_back(static_cast<uint8_t>(value & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 32) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 40) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 48) & 0xFF));
  data_.push_back(static_cast<uint8_t>((value >> 56) & 0xFF));
}

void ByteBuffer::write_f32(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(float));
  write_u32(bits);
}

void ByteBuffer::write_f64(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(double));
  write_u64(bits);
}

// read primitves
uint8_t ByteBuffer::read_u8(size_t& offset) const {
  return data_[offset++];
}

uint16_t ByteBuffer::read_u16(size_t& offset) const {
  uint16_t value = static_cast<uint16_t>(data_[offset]) |
                   (static_cast<uint16_t>(data_[offset + 1]) << 8);
  offset += 2;
  return value;
}

uint32_t ByteBuffer::read_u32(size_t& offset) const {
  uint32_t value = static_cast<uint32_t>(data_[offset]) |
                   (static_cast<uint32_t>(data_[offset + 1]) << 8) |
                   (static_cast<uint32_t>(data_[offset + 2]) << 16) |
                   (static_cast<uint32_t>(data_[offset + 3]) << 24);
  offset += 4;
  return value;
}

uint64_t ByteBuffer::read_u64(size_t& offset) const {
  uint64_t value = static_cast<uint64_t>(data_[offset]) |
                   (static_cast<uint64_t>(data_[offset + 1]) << 8) |
                   (static_cast<uint64_t>(data_[offset + 2]) << 16) |
                   (static_cast<uint64_t>(data_[offset + 3]) << 24) |
                   (static_cast<uint64_t>(data_[offset + 4]) << 32) |
                   (static_cast<uint64_t>(data_[offset + 5]) << 40) |
                   (static_cast<uint64_t>(data_[offset + 6]) << 48) |
                   (static_cast<uint64_t>(data_[offset + 7]) << 56);
  offset += 8;
  return value;
}

float ByteBuffer::read_f32(size_t& offset) const {
  uint32_t bits = read_u32(offset);
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

double ByteBuffer::read_f64(size_t& offset) const {
  uint64_t bits = read_u64(offset);
  double value;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

// static read methds for raw byte arrays
uint8_t ByteBuffer::read_u8(const uint8_t* data, size_t& offset) {
  return data[offset++];
}

uint16_t ByteBuffer::read_u16(const uint8_t* data, size_t& offset) {
  uint16_t value = static_cast<uint16_t>(data[offset]) |
                   (static_cast<uint16_t>(data[offset + 1]) << 8);
  offset += 2;
  return value;
}

uint32_t ByteBuffer::read_u32(const uint8_t* data, size_t& offset) {
  uint32_t value = static_cast<uint32_t>(data[offset]) |
                   (static_cast<uint32_t>(data[offset + 1]) << 8) |
                   (static_cast<uint32_t>(data[offset + 2]) << 16) |
                   (static_cast<uint32_t>(data[offset + 3]) << 24);
  offset += 4;
  return value;
}

uint64_t ByteBuffer::read_u64(const uint8_t* data, size_t& offset) {
  uint64_t value = static_cast<uint64_t>(data[offset]) |
                   (static_cast<uint64_t>(data[offset + 1]) << 8) |
                   (static_cast<uint64_t>(data[offset + 2]) << 16) |
                   (static_cast<uint64_t>(data[offset + 3]) << 24) |
                   (static_cast<uint64_t>(data[offset + 4]) << 32) |
                   (static_cast<uint64_t>(data[offset + 5]) << 40) |
                   (static_cast<uint64_t>(data[offset + 6]) << 48) |
                   (static_cast<uint64_t>(data[offset + 7]) << 56);
  offset += 8;
  return value;
}

float ByteBuffer::read_f32(const uint8_t* data, size_t& offset) {
  uint32_t bits = read_u32(data, offset);
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

double ByteBuffer::read_f64(const uint8_t* data, size_t& offset) {
  uint64_t bits = read_u64(data, offset);
  double value;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

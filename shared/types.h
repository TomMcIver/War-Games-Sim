#pragma once

#include <cstdint>

// entity 
using EntityId = uint64_t;

// team identifier
using Team = uint8_t;

// pod vector types
struct Vec2 {
  double x;
  double y;
};

struct Vec3 {
  double x;
  double y;
  double z;
};

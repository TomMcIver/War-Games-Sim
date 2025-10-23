#pragma once

#include "types.h"
#include "orders.h"
#include <deque>

// Position, orientaton, and floor level
struct Transform {
  Vec3 pos;
  float yaw;
  int floor;
};

// Velocity and movment constraints
struct Kinematics {
  Vec3 vel;
  float maxSpeed;
};

// Team affiliation
struct TeamTag {
  enum class Team : uint8_t {
    Red,
    Blue
  };
  Team team;
};

// order queue for entity
struct Orders {
  std::deque<Order> queue;
};

#pragma once

#include "types.h"

// Position, orientation, and floor level
struct Transform {
  Vec3 pos;
  float yaw;
  int floor;
};

// Velocity and movement constraints
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

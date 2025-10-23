#pragma once

class World;

class KinematicsSystem {
public:
  // Integrat velocity: pos = pos + vel * dt
  static void update(World& world, double dt);
};

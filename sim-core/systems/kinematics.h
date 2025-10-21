#pragma once

class World;

class KinematicsSystem {
public:
  // Integrate velocity: pos = pos + vel * dt
  static void update(World& world, double dt);
};

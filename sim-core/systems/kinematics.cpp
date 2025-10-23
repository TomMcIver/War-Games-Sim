#include "kinematics.h"
#include "sim-core/world.h"
#include <cmath>

void KinematicsSystem::update(World& world, double dt) {
  auto& registry = world.registry();

  auto view = registry.view<Transform, Kinematics>();
  for (auto entity : view) {
    auto& transform = view.get<Transform>(entity);
    auto& kinematics = view.get<Kinematics>(entity);

    // Clamp velcity to maxSpeed
    double speed = std::sqrt(kinematics.vel.x * kinematics.vel.x +
                            kinematics.vel.y * kinematics.vel.y +
                            kinematics.vel.z * kinematics.vel.z);

    if (speed > kinematics.maxSpeed) {
      double scale = kinematics.maxSpeed / speed;
      kinematics.vel.x *= scale;
      kinematics.vel.y *= scale;
      kinematics.vel.z *= scale;
    }

    // Integrat position
    transform.pos.x += kinematics.vel.x * dt;
    transform.pos.y += kinematics.vel.y * dt;
    transform.pos.z += kinematics.vel.z * dt;
  }
}

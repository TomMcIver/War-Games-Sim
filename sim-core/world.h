#pragma once

#include <entt/entt.hpp>
#include "shared/components.h"

class World {
public:
  World() = default;

  // Spawn entity with components
  EntityId spawn();

  // Despawn entity
  void despawn(EntityId id);

  // Access registry
  entt::registry& registry() { return registry_; }
  const entt::registry& registry() const { return registry_; }

private:
  entt::registry registry_;
};

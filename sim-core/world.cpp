#include "world.h"

EntityId World::spawn() {
  return static_cast<EntityId>(registry_.create());
}

void World::despawn(EntityId id) {
  registry_.destroy(static_cast<entt::entity>(id));
}

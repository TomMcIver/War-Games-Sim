#include "engine.h"
#include "world.h"
#include "systems/kinematics.h"
#include "shared/serialization.h"
#include "shared/components.h"
#include <entt/entt.hpp>

Engine::Engine(const SimConfig &config)
    : config_(config), currentTick_(0), processedCommandCount_(0), rng_(config.seed), world_(std::make_unique<World>()) {}

void Engine::enqueueUserCommand(const std::vector<uint8_t> &command) {
  commandQueue_.push_back(command);
}

void Engine::processOrders() {
  // Process queued commands
  processedCommandCount_ += commandQueue_.size();
  // TODO: Parse and execute commands
  commandQueue_.clear();
}

void Engine::tick() {
  // process queued commands
  processOrders();

  // run systems
  KinematicsSystem::update(*world_, config_.tickSeconds);

  //  simulation tick
  ++currentTick_;
}

void Engine::getStateSnapshot(std::vector<uint8_t> &snapshot) const {
  ByteBuffer buffer;

  // Version header: magic number + version
  buffer.write_u32(0x53494D53); // "SIMS" magic
  buffer.write_u16(1); // version 1

  // tick count
  buffer.write_u64(currentTick_);

  // entity count
  auto& registry = world_->registry();
  auto transformView = registry.view<Transform>();
  buffer.write_u32(static_cast<uint32_t>(transformView.size()));

  // serialize each entity with transform
  for (auto entity : transformView) {
    // Entity ID
    buffer.write_u64(static_cast<uint64_t>(entity));

    // transform component
    const auto& transform = transformView.get<Transform>(entity);
    buffer.write_f64(transform.pos.x);
    buffer.write_f64(transform.pos.y);
    buffer.write_f64(transform.pos.z);
    buffer.write_f32(transform.yaw);
    buffer.write_u32(static_cast<uint32_t>(transform.floor));

    // teamtag component 
    if (registry.all_of<TeamTag>(entity)) {
      buffer.write_u8(1); // has team tag
      const auto& teamTag = registry.get<TeamTag>(entity);
      buffer.write_u8(static_cast<uint8_t>(teamTag.team));
    } else {
      buffer.write_u8(0); // no team tag
    }
  }

  // buffer to snapshot
  snapshot.resize(buffer.size());
  std::memcpy(snapshot.data(), buffer.data(), buffer.size());
}

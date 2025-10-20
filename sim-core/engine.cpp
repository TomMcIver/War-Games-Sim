#include "engine.h"

Engine::Engine(const SimConfig &config)
    : config_(config), currentTick_(0), rng_(config.seed) {}

void Engine::enqueueUserCommand(const std::vector<uint8_t> &command) {
  commandQueue_.push_back(command);
}

void Engine::tick() {
  // process queued commands
  commandQueue_.clear();

  //  simulation tick
  ++currentTick_;
}

void Engine::getStateSnapshot(std::vector<uint8_t> &snapshot) const {
  snapshot.clear();
  //  tick count for now
  snapshot.resize(sizeof(uint64_t));
  const auto *tickPtr = reinterpret_cast<const uint8_t *>(&currentTick_);
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    snapshot[i] = tickPtr[i];
  }
}

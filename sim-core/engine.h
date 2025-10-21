#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

class World;

struct SimConfig {
  double tickSeconds;
  uint64_t seed;
};

class Engine {
public:
  explicit Engine(const SimConfig &config);

  void enqueueUserCommand(const std::vector<uint8_t> &command);
  void tick();
  void getStateSnapshot(std::vector<uint8_t> &snapshot) const;

  uint64_t getCurrentTick() const { return currentTick_; }
  World& getWorld() { return *world_; }
  const World& getWorld() const { return *world_; }

private:
  SimConfig config_;
  uint64_t currentTick_;
  std::mt19937_64 rng_;
  std::vector<std::vector<uint8_t>> commandQueue_;
  std::unique_ptr<World> world_;
};

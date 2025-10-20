#include "sim-core/engine.h"
#include "sim-core/tick_loop.h"
#include <iostream>

int main() {
  SimConfig config{0.016, 12345}; // 60 fps, seed 12345
  Engine engine(config);

  for (int i = 0; i < 100; ++i) {
    TickLoop::runFixed(engine, config.tickSeconds);
    std::cout << "tick " << engine.getCurrentTick() << "\n";
  }

  return 0;
}

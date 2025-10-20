#include "tick_loop.h"
#include "engine.h"

void TickLoop::runFixed(Engine &engine, double frameDelta) {

  (void)frameDelta; 
  engine.tick();
}

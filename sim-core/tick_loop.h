#pragma once

class Engine;

class TickLoop {
public:
  static void runFixed(Engine &engine, double frameDelta);
};

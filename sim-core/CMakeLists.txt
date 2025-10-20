add_library(simcore STATIC
  engine.cpp
  tick_loop.cpp
)

target_include_directories(simcore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

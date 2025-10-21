add_library(simcore STATIC
  engine.cpp
  tick_loop.cpp
  world.cpp
  systems/kinematics.cpp
)

target_include_directories(simcore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(simcore PUBLIC
  EnTT::EnTT
)

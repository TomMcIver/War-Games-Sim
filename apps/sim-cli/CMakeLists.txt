add_executable(sim-cli
  main.cpp
)

target_link_libraries(sim-cli PRIVATE simcore)

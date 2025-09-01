cmake_minimum_required(VERSION 3.10)
project(AutogradProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directory for headers (value.h, doctest.h, etc.)
include_directories(include)

# --- Library (core) ---
add_library(core
    core/value.cpp
    core/neuron.cpp
    core/layer.cpp
    core/nn.cpp
    core/loss.cpp
    core/checkpoint.cpp
    core/util.cpp
)

target_include_directories(core PUBLIC include)

# --- Main executable ---
add_executable(main main.cpp)
target_link_libraries(main PRIVATE core)

add_executable(experiment_spiral_train experiments/spiral/train.cpp)
target_link_libraries(experiment_spiral_train PRIVATE core)

add_executable(experiment_spiral_test experiments/spiral/test.cpp)
target_link_libraries(experiment_spiral_test PRIVATE core)

# --- Tests ---
enable_testing()

add_executable(run_value_test tests/test_value.cpp)

target_link_libraries(run_value_test PRIVATE core)

add_test(NAME ValueTest COMMAND run_value_test)

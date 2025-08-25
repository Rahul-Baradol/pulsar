cmake_minimum_required(VERSION 3.10)
project(AutogradProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directory for headers (value.h, doctest.h, etc.)
include_directories(include)

# --- Library (core) ---
add_library(core
    core/value.cpp
    core/neuron.cpp
    core/layer.cpp
    core/nn.cpp
)

target_include_directories(core PUBLIC include)

# --- Main executable ---
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE core)

add_executable(data_spiral_predict data/spiral/predict.cpp)
target_link_libraries(data_spiral_predict PRIVATE core)

# --- Tests ---
enable_testing()

add_executable(run_value_test tests/test_value.cpp)
add_executable(run_nn_test tests/test_nn.cpp)

target_link_libraries(run_value_test PRIVATE core)
target_link_libraries(run_nn_test PRIVATE core)

add_test(NAME ValueTest COMMAND run_value_test)
add_test(NAME NNTest COMMAND run_nn_test)

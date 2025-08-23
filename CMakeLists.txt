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

# --- Tests ---
enable_testing()

add_executable(runTests tests/test_value.cpp)
target_link_libraries(runTests PRIVATE core)

add_test(NAME AutogradTests COMMAND runTests)

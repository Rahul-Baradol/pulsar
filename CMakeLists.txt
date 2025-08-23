cmake_minimum_required(VERSION 3.10)
project(AutogradProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directory for headers (value.h, doctest.h, etc.)
include_directories(include)

# --- Library (autograd) ---
add_library(autograd
    autograd/value.cpp
)

target_include_directories(autograd PUBLIC include)

# --- Main executable ---
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE autograd)

# --- Tests ---
enable_testing()

add_executable(runTests tests/test_value.cpp)
target_link_libraries(runTests PRIVATE autograd)

add_test(NAME AutogradTests COMMAND runTests)

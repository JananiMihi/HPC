cmake_minimum_required(VERSION 3.10)
project(RecommendationSystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 1: Serial Baseline
add_executable(phase1_serial main.cpp recommendation.cpp)

# Compiler flags for optimization
if(MSVC)
    target_compile_options(phase1_serial PRIVATE /O2 /W4)
else()
    target_compile_options(phase1_serial PRIVATE -O2 -Wall -Wextra)
endif()

# Set output directory
set_target_properties(phase1_serial PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

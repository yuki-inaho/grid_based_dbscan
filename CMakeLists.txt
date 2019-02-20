cmake_minimum_required(VERSION 2.8)
project(grid_based_dbscan)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-O0 -std=gnu++11 -fopenmp -fexceptions")
add_executable(grid_based_dbscan src/main.cpp src/grid_based_dbscan.cpp)

include_directories(
    include
)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(grid_based_dbscan ${PCL_LIBRARIES})

cmake_minimum_required(VERSION 3.25)
project(visual_odometry)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(apps)
add_subdirectory(src)
add_subdirectory(tests)

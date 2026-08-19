cmake_minimum_required(VERSION 3.16.3)
project(rt_torque_bridge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Point CMAKE_PREFIX_PATH at the RDK install prefix, e.g.
#   cmake .. -DCMAKE_PREFIX_PATH=~/rdk_install
find_package(flexiv_rdk REQUIRED)

add_executable(rt_torque_bridge rt_torque_bridge.cpp)
target_link_libraries(rt_torque_bridge flexiv::flexiv_rdk rt pthread)

cmake_minimum_required(VERSION 3.24...4.4)

project(lac_minorminer VERSION 0.1.0 LANGUAGES CXX)

option(LAC_MINORMINER_BUILD_PYTHON "Build the pybind11 extension" ON)
option(BUILD_TESTING "Build the native CTest suite" OFF)

include(CTest)
add_subdirectory(cpp)

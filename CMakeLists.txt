cmake_minimum_required(VERSION 3.9)
project(EMIDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/option)

add_executable(emida src/main.cpp)


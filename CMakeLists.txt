cmake_minimum_required(VERSION 3.9)
project(EMIDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(GoogleTest)
# Download and unpack googletest at configure time
configure_file(gtest_external.cmake googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

enable_testing()



find_package(CUDA REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION On)


include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/option)


file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cu")
file(GLOB_RECURSE TEST_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "test/*.cpp")

list(REMOVE_ITEM CPP_SOURCES "src/main.cpp")

add_executable(emida
    src/main.cpp
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)


add_executable(emida_test
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${TEST_SOURCES}
)

add_custom_target(emida_test_data_copy ALL
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/test/res ${CMAKE_BINARY_DIR}/test/res/)
add_dependencies(emida_test emida_test_data_copy)

target_include_directories(emida_test PUBLIC ${CMAKE_SOURCE_DIR}/src/)

target_link_libraries(emida_test gtest_main)
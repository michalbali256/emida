cmake_minimum_required(VERSION 3.9)
project(EMIDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(ExternalProject)

if(UNIX)
  find_package(ZLIB REQUIRED)
  find_package(LibLZMA REQUIRED)
endif()

ExternalProject_Add(tiff_project
  GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
  GIT_TAG        v4.1.0
  GIT_SHALLOW    On
  PREFIX         externals/libtiff/
  LOG_CONFIGURE ON
  LOG_BUILD ON
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DBUILD_SHARED_LIBS=Off -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  INSTALL_COMMAND COMMAND "${CMAKE_COMMAND}" -P cmake_install.cmake
  BUILD_BYPRODUCTS ${TIFF_STATIC_LIBS}
)

link_directories(${CMAKE_BINARY_DIR}/externals/libtiff/lib ${CMAKE_BINARY_DIR}/externals/libtiff/lib64)
include_directories(${CMAKE_BINARY_DIR}/externals/libtiff/include)


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
include_directories(${CMAKE_SOURCE_DIR}/option)


set(FILESYSTEM_LIB "")
if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9"))
    set(FILESYSTEM_LIB "stdc++fs")
endif()

file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cu")
file(GLOB_RECURSE TEST_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "test/*.cpp")

list(REMOVE_ITEM CPP_SOURCES "src/main.cpp")

add_library(emida_lib STATIC
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)
add_dependencies(emida_lib tiff_project)
target_link_libraries(emida_lib
    debug tiffd
    optimized tiff
    ${FILESYSTEM_LIB}
)
if(UNIX)
    target_link_libraries(emida_lib
        ${ZLIB_LIBRARIES}
        ${LIBLZMA_LIBRARIES}
    )
endif()

target_compile_options(emida_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=compute_60>)


add_executable(emida
    src/main.cpp
)
target_link_libraries(emida emida_lib)


add_executable(emida_test
    ${TEST_SOURCES}
)
target_link_libraries(emida_test emida_lib
    debug tiffd
    optimized tiff
)


add_custom_target(emida_test_data_copy ALL
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/test/res ${CMAKE_SOURCE_DIR}/bin/test/res/)
add_dependencies(emida_test emida_test_data_copy)

target_include_directories(emida_test PUBLIC ${CMAKE_SOURCE_DIR}/src/)

target_link_libraries(emida_test gtest_main)

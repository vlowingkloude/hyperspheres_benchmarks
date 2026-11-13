cmake_minimum_required(VERSION 4.2)

project(Hyperspheres)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)

OPTION(BUILD_EXAMPLE "build the main file in example" OFF)

add_subdirectory(definitions)
add_subdirectory(memorymap)
add_subdirectory(distance)
add_subdirectory(storage)
add_subdirectory(execution)
add_subdirectory(database)

add_subdirectory(interface)

if (BUILD_EXAMPLE)
    add_subdirectory(example)
endif ()

add_subdirectory(main)

add_library(
        Hyperspheres_build_options INTERFACE
)
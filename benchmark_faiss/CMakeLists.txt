cmake_minimum_required(VERSION 4.2)

project(spheres_benchmark)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(faiss)

add_executable(
        spheres_benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(spheres_benchmark
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
        spheres_benchmark
        PRIVATE
        faiss_benchmark
)

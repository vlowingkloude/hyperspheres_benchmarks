add_library(memorymap)

target_sources(
        memorymap
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/memorymap.cppm
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/memorymap.cpp
)

target_link_libraries(
        memorymap
        PRIVATE
        Hyperspheres_build_options
)
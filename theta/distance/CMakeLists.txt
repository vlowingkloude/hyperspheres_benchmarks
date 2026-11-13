add_library(distance)

target_sources(
        distance
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/distance.cppm
)

target_link_libraries(
        distance
        PRIVATE
        m
        Hyperspheres_build_options
)
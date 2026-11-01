add_library(hyperspheres)

target_sources(
        hyperspheres
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/interface.cppm
)

target_link_libraries(
        hyperspheres
        PRIVATE
        database
        definitions
        distance
        execution
        memorymap
        storage
        m
        Hyperspheres_build_options
)
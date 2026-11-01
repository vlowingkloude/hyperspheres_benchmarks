add_library(definitions)

target_sources(
        definitions
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/definitions.cppm
)

target_link_libraries(
        definitions
        PRIVATE
        Hyperspheres_build_options
)
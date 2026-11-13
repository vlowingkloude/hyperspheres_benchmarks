add_library(execution)

target_sources(
        execution
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/execution.cppm
)

target_link_libraries(
        execution
        PRIVATE
        definitions
        storage
        Hyperspheres_build_options
)
add_library(database)

target_sources(
        database
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/database.cppm
)

target_link_libraries(
        database
        PRIVATE
        definitions
        storage
        execution
        distance
        Hyperspheres_build_options
)
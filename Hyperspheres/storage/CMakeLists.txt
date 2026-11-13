add_library(storage)

target_sources(
        storage
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/storage.cppm
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/data_file.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/page.cpp
)

target_link_libraries(
        storage
        PRIVATE
        memorymap
        Hyperspheres_build_options
)
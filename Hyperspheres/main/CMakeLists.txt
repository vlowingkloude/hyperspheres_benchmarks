add_executable(
        main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(main
        PRIVATE
        hyperspheres
        Hyperspheres_build_options
)
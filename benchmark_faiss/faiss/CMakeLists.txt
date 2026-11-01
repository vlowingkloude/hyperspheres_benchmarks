include(FetchContent)

set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        faiss_source
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG v1.12.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(faiss_source)

add_library(faiss_benchmark faiss_benchmark.cpp)
target_include_directories(faiss_benchmark PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(faiss_benchmark PRIVATE faiss)

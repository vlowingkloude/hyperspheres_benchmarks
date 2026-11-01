module;

#include <cstdint>
#include <compare>
#include <string>

export module definitions;

export namespace hyperspheres::defaults {

    // 256MB each file by default
    // each file contains n pages
    // metadata includes: page_size, type_length, type of vector elements, measure_length, distance measure
    //                    and longest_acceptable_distance_in_a_page
    constexpr std::size_t default_single_file_size {256 * 1024 * 1024};

    // a file contains pages, which is 8MB by default
    // for a common data page, the structure is:
    //      n, T max_dist, [id], [vectors]
    constexpr std::size_t default_page_size {8 * 1024 * 1024};

    // config file
    const std::string config_filename {"hyperspheres.config"};

    // max number of a bytes of NativeVectorElementType, 8 bytes
    constexpr std::size_t max_n_bytes_of_native_type {8};

    constexpr std::size_t first_n_bytes_per_page {8 + max_n_bytes_of_native_type};

    constexpr double density_threshold_of_a_file {0.67};

    constexpr std::size_t default_thread_local_buffer_size {4096};

}

export namespace hyperspheres::supported {

    enum class SupportedDataType : uint8_t {
        Float
    };

    enum class SupportedDistanceFunctions : uint8_t {
        SquaredL2Distance
    };

    enum class PruningMode : uint8_t {
        Safe,
        Aggressive
    };

}


export namespace hyperspheres::structures {
    template <typename ElemT, typename IdT = std::size_t>
        requires std::three_way_comparable<ElemT> && std::three_way_comparable<IdT>
    using IdTaggedDistance = std::pair<ElemT, IdT>;

    struct DatabaseConfig {
        hyperspheres::supported::SupportedDataType dtype {supported::SupportedDataType::Float};
        hyperspheres::supported::SupportedDistanceFunctions distance_function {supported::SupportedDistanceFunctions::SquaredL2Distance};
        std::size_t page_size {defaults::default_page_size};
        std::size_t file_size {defaults::default_single_file_size};
        std::size_t dim {1024};
        std::size_t n_files {0};
    };

}
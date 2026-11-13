#ifndef SPHERES_BENCHMARK_BENCHMARK_CONFIG_H
#define SPHERES_BENCHMARK_BENCHMARK_CONFIG_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <span>
#include <algorithm>
#include <bit>
#include <limits>
#include <cstdlib>
#include <charconv>
#include <system_error>

struct ScopedTimer {
    using Clock = std::chrono::steady_clock;

    Clock::time_point start = Clock::now();
    double& result_ms;

    explicit ScopedTimer(double& result)
        : result_ms(result) {}

    ~ScopedTimer() noexcept {
        result_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - start
        ).count();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;
};

enum class DatasetType : uint8_t {
    GIST1M,
    SIFT1M,
    Generated
};

enum class DistanceType : uint8_t {
    NormalizedL2,
    SquaredNormalizedL2,
    SquaredL2
};

constexpr std::size_t M = 16;
constexpr std::size_t efConstruction = 128;
constexpr std::size_t efSearch = 200;

struct BenchmarkConfig {
    std::string engine_name;
    std::string engine_mode;

    std::size_t n_threads = 4;

    DatasetType dataset_type = DatasetType::SIFT1M;
    std::string dataset_path;

    DistanceType distance_type = DistanceType::NormalizedL2;

    std::size_t vector_dim = 0;
    std::size_t n_base_vectors = 0;
    std::size_t n_query_vectors = 0;
    std::size_t n_learn_vectors = 0;
    std::size_t k = 0;
    std::size_t k_in_gt = 0;

    std::string metrics_path;
    std::string index_path;

    bool reuse_index = false;

    // inferred args
    std::string base_vector_path;
    std::string query_vector_path;
    std::string gt_vector_path;
    std::string learn_vector_path;

    // engine specific configs

    // spheres
    std::size_t spheres_page_size = 8 * 1024 * 1024;
    std::size_t spheres_file_size = 256 * 1024 * 1024;
    std::size_t spheres_n_pages_to_probe = 0;
    std::size_t spheres_local_buffer_size = 30000;
    std::size_t spheres_morsel_step = 1;

    // faiss
    std::size_t faiss_hnsw_m = M;
    std::size_t faiss_hnsw_efsearch = efSearch;
    std::size_t faiss_hnsw_efconstruction = efConstruction;
    std::size_t faiss_lsh_nbits = 256;
    std::size_t faiss_ivfflat_nlists = 4000;
    std::size_t faiss_ivfflat_nprobe = 100;
    std::size_t faiss_ivfflat_ingest_batch_size = 10000;
};

[[nodiscard]]
inline DatasetType parse_dataset_type(const std::string_view value) {
    if (value == "sift1m") {
        return DatasetType::SIFT1M;
    }
    if (value == "gist1m") {
        return DatasetType::GIST1M;
    }
    return DatasetType::Generated;
}

[[nodiscard]]
inline DistanceType parse_distance_type(const std::string_view value) {
    if (value == "norml2") {
        return DistanceType::NormalizedL2;
    }
    if (value == "squared_norml2") {
        return DistanceType::SquaredNormalizedL2;
    }
    if (value == "squared_l2") {
        return DistanceType::SquaredL2;
    }
    throw std::invalid_argument(std::string("invalid value for ") + std::string(value));
}

inline void make_benchmark_with_default_params(BenchmarkConfig& config) {
    switch (config.dataset_type) {
        case DatasetType::GIST1M:
            config.k = 100;
            config.k_in_gt = 100;
            config.vector_dim = 960;
            config.n_base_vectors = 1000000;
            config.n_query_vectors = 1000;
            config.n_learn_vectors = 500000;
            break;
        case DatasetType::SIFT1M:
            config.k = 100;
            config.k_in_gt = 100;
            config.vector_dim = 128;
            config.n_base_vectors = 1000000;
            config.n_query_vectors = 10000;
            config.n_learn_vectors = 100000;
            break;
        default:
            config.k = 100;
            config.k_in_gt = 100;
            config.vector_dim = 128;
            config.n_base_vectors = 10000000;
            config.n_query_vectors = 10000;
            config.n_learn_vectors = 100000;
    }
}

inline std::vector<std::vector<float>> read_file_as_vectors(const std::string& path, std::size_t rows, std::size_t dim) {
    std::vector<std::vector<float>> data(rows);
    std::ifstream ins(path, std::ios::binary);
    if (!ins) {
        throw std::runtime_error("failed to open " + path);
    }
    int32_t d;
    for (auto i = 0; i < rows; ++i) {
        std::vector<float> temp (dim);
        ins.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!ins || d != static_cast<std::int32_t>(dim)) {
            throw std::runtime_error("invalid dimension or truncated file: " + path);
        }
        ins.read(std::bit_cast<char *, float *>(temp.data()), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!ins) {
            throw std::runtime_error("truncated vector data: " + path);
        }
        data[i] = std::move(temp);
    }
    return data;
}

inline std::vector<std::vector<int>> read_gt_vectors(const std::string& path, std::size_t rows, std::size_t dim) {
    std::vector<std::vector<int>> data(rows);
    std::ifstream ins(path, std::ios::binary);
    if (!ins) {
        throw std::runtime_error("failed to open " + path);
    }
    int32_t d;
    for (auto i = 0; i < rows; ++i) {
        std::vector<int> temp (dim);
        ins.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!ins || d != static_cast<std::int32_t>(dim)) {
            throw std::runtime_error("invalid dimension or truncated file: " + path);
        }
        ins.read(std::bit_cast<char *, int *>(temp.data()), static_cast<std::streamsize>(dim * sizeof(int)));
        if (!ins) {
            throw std::runtime_error("truncated ground truth: " + path);
        }
        data[i] = std::move(temp);
    }
    return data;
}

inline std::vector<float> read_file_as_flat_array(const std::string& path, std::size_t rows, std::size_t dim) {
    std::vector<float> data(rows * dim);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    int32_t d;
    for (auto i = 0; i < rows; ++i) {
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!in || d != static_cast<std::int32_t>(dim)) {
            throw std::runtime_error("invalid dimension or truncated file: " + path);
        }
        in.read(std::bit_cast<char *, float *>(data.data() + i * dim), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!in) {
            throw std::runtime_error("truncated vector data: " + path);
        }
    }
    return data;
}

inline double squared_l2_distance(std::span<const float> x, std::span<const float> y) {
    const double l2 = std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<double> {},
        [](const float ix, const float iy) {
            const double diff = static_cast<double>(ix) - static_cast<double>(iy);
            return diff * diff;
        });
    return std::max(0.0, l2);
}

inline std::pair<double, double> avg_squared_l2_distance(
                   std::span<const float> query,
                   std::span<const float> base,
                   std::size_t dim,
                   const std::vector<std::vector<std::size_t>>& results,
                   const std::vector<std::vector<int>>& gt,
                   const std::size_t k) {
    if (results.empty() || results.size() != gt.size() ||
        query.size() != results.size() * dim || dim == 0 || base.size() % dim != 0) {
        throw std::invalid_argument("inconsistent inputs for average squared L2 distance");
    }
    const std::size_t base_rows = base.size() / dim;
    double r1 = 0.0;
    double r2 = 0.0;

    for (std::size_t q = 0; q < results.size(); ++q) {
        if (results[q].size() < k || gt[q].size() < k) {
            throw std::invalid_argument("result or ground-truth row contains fewer than k IDs");
        }
        double qri = 0.0;
        double gri = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            if (results[q][i] >= base_rows || gt[q][i] < 0 ||
                static_cast<std::size_t>(gt[q][i]) >= base_rows) {
                throw std::out_of_range("result or ground-truth vector ID is out of range");
            }
            const std::span<const float> query_vector(query.data() + q * dim, dim);
            const std::span<const float> result_vector(base.data() + results[q][i] * dim, dim);
            const std::span<const float> truth_vector(base.data() + static_cast<std::size_t>(gt[q][i]) * dim, dim);
            qri += squared_l2_distance(result_vector, query_vector);
            gri += squared_l2_distance(truth_vector, query_vector);
        }
        r1 += (qri / static_cast<double>( k));
        r2 += (gri / static_cast<double>( k));
    }

    return std::make_pair(r1 / static_cast<double>(results.size()), r2 / static_cast<double>(results.size()));
}

inline double recall(const std::vector<std::vector<std::size_t>>& results,
                   const std::vector<std::vector<int>>& gt,
                   const std::size_t k) {
    if (results.empty() || results.size() != gt.size() || k == 0) {
        throw std::invalid_argument("inconsistent inputs for recall");
    }
    double total = 0.0;
    for (auto i = 0; i < results.size(); ++i) {
        if (results[i].size() != k || gt[i].size() < k) {
            throw std::invalid_argument("each result row must contain exactly k IDs");
        }
        std::unordered_set<int> truth(gt[i].begin(), gt[i].begin() + static_cast<long>(k));
        std::size_t hits = 0;
        for (const auto& id : results[i]) {
            if (truth.contains(static_cast<int>(id))) {
                ++hits;
            }
        }
        total += static_cast<double>(hits) / static_cast<double>(k);
    }
    return total / static_cast<double>(results.size());
}

inline std::string quote(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped.push_back(hex[character >> 4]);
                    escaped.push_back(hex[character & 0x0f]);
                } else {
                    escaped.push_back(static_cast<char>(character));
                }
        }
    }
    escaped.push_back('"');
    return escaped;
}

[[nodiscard]]
inline std::size_t parse_size(const std::string_view value, const std::string_view option) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid unsigned integer for --" + std::string(option));
    }
    return result;
}

inline void write_json(const std::string& path, const std::vector<std::pair<std::string, std::string>>& fields) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path);
    }
    out << "{\n";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        out << "  \"" << fields[i].first << "\": " << fields[i].second;
        out << (i + 1 == fields.size() ? "\n" : ",\n");
    }
    out << "}\n";
}

inline BenchmarkConfig parse_argv(int argc, char** argv) {
    std::unordered_map<std::string, std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc / 2));

    for (int i = 1; i < argc; i += 2) {
        const std::string_view option = argv[i];
        if (!option.starts_with("--")) {
            throw std::invalid_argument(
                "expected an option beginning with --, got: "
                + std::string(option));
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "missing value for option: " + std::string(option));
        }
        const std::string key(option.substr(2));
        const std::string value(argv[i + 1]);
        if (!arguments.emplace(key, value).second) {
            throw std::invalid_argument(
                "duplicate option: --" + key);
        }
    }

    BenchmarkConfig config;
    const auto dataset = arguments.find("dataset");
    if (dataset == arguments.end()) {
        throw std::invalid_argument("missing option: --dataset");
    }
    const auto dataset_type = parse_dataset_type(dataset->second);
    config.dataset_type = dataset_type;
    make_benchmark_with_default_params(config);

    for (const auto & [key, value] : arguments) {
        if (key == "dataset") {}
        else if (key == "engine") {config.engine_name = value;}
        else if (key == "mode") {config.engine_mode = value;}
        else if (key == "n_threads") {config.n_threads = parse_size(value, key);}
        else if (key == "path") {config.dataset_path = value;}
        else if (key == "distance") {config.distance_type = parse_distance_type(value);}
        else if (key == "dim") {config.vector_dim = parse_size(value, key);}
        else if (key == "n_base") {config.n_base_vectors = parse_size(value, key);}
        else if (key == "n_learn") {config.n_learn_vectors = parse_size(value, key);}
        else if (key == "n_query") {config.n_query_vectors = parse_size(value, key);}
        else if (key == "k") {config.k = parse_size(value, key);}
        else if (key == "kgt") {config.k_in_gt = parse_size(value, key);}
        else if (key == "metrics") {config.metrics_path = value;}
        else if (key == "index") {config.index_path = value;}
        else if (key == "reuse_index") {config.reuse_index = value == "1" ? true : false;}
        else if (key == "spheres_page_size") {config.spheres_page_size = parse_size(value, key);}
        else if (key == "spheres_file_size") {config.spheres_file_size = parse_size(value, key);}
        else if (key == "spheres_n_ppage") {config.spheres_n_pages_to_probe = parse_size(value, key);}
        else if (key == "spheres_buf_size") {config.spheres_local_buffer_size = parse_size(value, key);}
        else if (key == "spheres_morsel") {config.spheres_morsel_step = parse_size(value, key);}

        else if (key == "faiss_hnsw_m") {config.faiss_hnsw_m = parse_size(value, key);}
        else if (key == "faiss_hnsw_search") {config.faiss_hnsw_efsearch = parse_size(value, key);}
        else if (key == "faiss_hnsw_con") {config.faiss_hnsw_efconstruction = parse_size(value, key);}
        else if (key == "faiss_lsh_nb") {config.faiss_lsh_nbits = parse_size(value, key);}
        else if (key == "faiss_ivf_nlists") {config.faiss_ivfflat_nlists = parse_size(value, key);}
        else if (key == "faiss_ivf_nprobe") {config.faiss_ivfflat_nprobe = parse_size(value, key);}
        else if (key == "faiss_ivf_batch") {config.faiss_ivfflat_ingest_batch_size = parse_size(value, key);}
        else {throw std::invalid_argument("unknown option: --" + key);}
    }

    switch (dataset_type) {
        case DatasetType::GIST1M:
            config.base_vector_path = (std::filesystem::path(config.dataset_path) / "gist_base.fvecs").string();
            config.query_vector_path = (std::filesystem::path(config.dataset_path) / "gist_query.fvecs").string();
            config.gt_vector_path = (std::filesystem::path(config.dataset_path) / "gist_groundtruth.ivecs").string();
            config.learn_vector_path = (std::filesystem::path(config.dataset_path) / "gist_learn.fvecs").string();
            break;
        case DatasetType::SIFT1M:
            config.base_vector_path = (std::filesystem::path(config.dataset_path) / "sift_base.fvecs").string();
            config.query_vector_path = (std::filesystem::path(config.dataset_path) / "sift_query.fvecs").string();
            config.gt_vector_path = (std::filesystem::path(config.dataset_path) / "sift_groundtruth.ivecs").string();
            config.learn_vector_path = (std::filesystem::path(config.dataset_path) / "sift_learn.fvecs").string();
            break;
        default:
            config.base_vector_path = config.dataset_path + "base.fvecs";
            config.query_vector_path = config.dataset_path + "query.fvecs";
            config.gt_vector_path = config.dataset_path + "groundtruth.ivecs";
            config.learn_vector_path = config.dataset_path + "learn.fvecs";
            break;
    }
    if (config.engine_name.empty()) {
        throw std::invalid_argument("missing option: --engine");
    }
    if (config.dataset_path.empty()) {
        throw std::invalid_argument("missing option: --path");
    }
    if (config.metrics_path.empty()) {
        throw std::invalid_argument("missing option: --metrics");
    }
    if (config.engine_name != "baseline" && config.engine_name != "pgvector" &&
        config.index_path.empty()) {
        throw std::invalid_argument("missing option: --index");
    }
    if (config.n_threads == 0 || config.vector_dim == 0 || config.n_base_vectors == 0 ||
        config.n_query_vectors == 0 || config.k == 0 || config.k > config.k_in_gt ||
        config.k > config.n_base_vectors) {
        throw std::invalid_argument("invalid zero/count/k benchmark configuration");
    }
    if (config.distance_type != DistanceType::SquaredL2) {
        throw std::invalid_argument("current engine harnesses require --distance squared_l2");
    }

    return config;
}

#endif //SPHERES_BENCHMARK_BENCHMARK_CONFIG_H

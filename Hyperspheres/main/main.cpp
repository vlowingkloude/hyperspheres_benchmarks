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

#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

import hyperspheres;

namespace
{
    enum class DatasetType : uint8_t
    {
        GIST1M,
        SIFT1M,
        Generated
    };

    struct BenchmarkConfig
    {
        std::string engine_name;
        std::string engine_mode;

        std::size_t n_threads = 4;

        DatasetType dataset_type = DatasetType::SIFT1M;
        std::string dataset;
        std::string dataset_path;

        std::size_t vector_dim = 0;
        std::size_t n_base_vectors = 0;
        std::size_t n_query_vectors = 0;
        std::size_t n_learn_vectors = 0;
        std::size_t k = 0;
        std::size_t k_in_gt = 0;

        std::string result;

        bool reuse_index = false;

        // inferred args
        std::string base_vector_path;
        std::string query_vector_path;
        std::string gt_vector_path;
        std::string learn_vector_path;

        // engine specific configs

        // spheres
        std::size_t hs_page_size = 8 * 1024 * 1024;
        std::size_t hs_file_size = 256 * 1024 * 1024;
        std::size_t hs_n_pages_to_probe = 0;
        std::size_t hs_local_buffer_size = 30000;
        std::size_t hs_morsel_step = 1;
        float hs_beta = 1.0f;
    };

    struct ScopedTimer
    {
        using Clock = std::chrono::steady_clock;

        Clock::time_point start = Clock::now();
        double& result_ms;

        explicit ScopedTimer(double& result)
            : result_ms(result)
        {
        }

        ~ScopedTimer() noexcept
        {
            result_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - start
            ).count();
        }

        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;
        ScopedTimer(ScopedTimer&&) = delete;
        ScopedTimer& operator=(ScopedTimer&&) = delete;
    };

    inline DatasetType parse_dataset_type(const std::string_view value)
    {
        if (value == "sift1m")
        {
            return DatasetType::SIFT1M;
        }
        if (value == "gist1m")
        {
            return DatasetType::GIST1M;
        }
        return DatasetType::Generated;
    }

    std::vector<std::vector<int>> read_gt_vectors(const std::string& path, std::size_t rows, std::size_t dim)
    {
        std::vector<std::vector<int>> data(rows);
        std::ifstream ins(path, std::ios::binary);
        if (!ins)
        {
            throw std::runtime_error("failed to open " + path);
        }
        int32_t d;
        for (auto i = 0; i < rows; ++i)
        {
            std::vector<int> temp(dim);
            ins.read(reinterpret_cast<char*>(&d), sizeof(d));
            if (!ins || d != static_cast<std::int32_t>(dim))
            {
                throw std::runtime_error("invalid dimension or truncated file: " + path);
            }
            ins.read(std::bit_cast<char*, int*>(temp.data()), static_cast<std::streamsize>(dim * sizeof(int)));
            if (!ins)
            {
                throw std::runtime_error("truncated ground truth: " + path);
            }
            data[i] = std::move(temp);
        }
        return data;
    }

    double recall(const std::vector<std::vector<std::size_t>>& results,
                  const std::vector<std::vector<int>>& gt,
                  const std::size_t k)
    {
        if (results.empty() || results.size() != gt.size() || k == 0)
        {
            throw std::invalid_argument("inconsistent inputs for recall");
        }
        double total = 0.0;
        for (auto i = 0; i < results.size(); ++i)
        {
            if (results[i].size() != k || gt[i].size() < k)
            {
                throw std::invalid_argument("each result row must contain exactly k IDs");
            }
            std::unordered_set<int> truth(gt[i].begin(), gt[i].begin() + static_cast<long>(k));
            std::size_t hits = 0;
            std::unordered_set<std::size_t> result(results[i].begin(), results[i].end());
            for (const auto& id : result)
            {
                if (truth.contains(static_cast<int>(id)))
                {
                    ++hits;
                }
            }
            total += static_cast<double>(hits) / static_cast<double>(k);
        }
        return total / static_cast<double>(results.size());
    }

    std::vector<std::vector<float>> read_file_as_vectors(const std::string& path, std::size_t rows, std::size_t dim)
    {
        std::vector<std::vector<float>> data(rows);
        std::ifstream ins(path, std::ios::binary);
        if (!ins)
        {
            throw std::runtime_error("failed to open " + path);
        }
        int32_t d;
        for (auto i = 0; i < rows; ++i)
        {
            std::vector<float> temp(dim);
            ins.read(reinterpret_cast<char*>(&d), sizeof(d));
            if (!ins || d != static_cast<std::int32_t>(dim))
            {
                throw std::runtime_error("invalid dimension or truncated file: " + path);
            }
            ins.read(std::bit_cast<char*, float*>(temp.data()), static_cast<std::streamsize>(dim * sizeof(float)));
            if (!ins)
            {
                throw std::runtime_error("truncated vector data: " + path);
            }
            data[i] = std::move(temp);
        }
        return data;
    }

    inline std::vector<float> read_file_as_flat_array(const std::string& path, std::size_t rows, std::size_t dim)
    {
        std::vector<float> data(rows * dim);
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("failed to open " + path);
        }
        int32_t d;
        for (auto i = 0; i < rows; ++i)
        {
            in.read(reinterpret_cast<char*>(&d), sizeof(d));
            if (!in || d != static_cast<std::int32_t>(dim))
            {
                throw std::runtime_error("invalid dimension or truncated file: " + path);
            }
            in.read(std::bit_cast<char*, float*>(data.data() + i * dim),
                    static_cast<std::streamsize>(dim * sizeof(float)));
            if (!in)
            {
                throw std::runtime_error("truncated vector data: " + path);
            }
        }
        return data;
    }

    inline std::vector<float> recall_summarize(const std::vector<std::vector<std::size_t>>& results,
                                               const std::vector<std::vector<int>>& gt,
                                               const std::size_t k)
    {
        if (results.empty() || results.size() != gt.size() || k == 0)
        {
            throw std::invalid_argument("inconsistent inputs for recall");
        }
        double total = 0.0;
        std::vector<float> recall_vec(results.size());
        for (auto i = 0; i < results.size(); ++i)
        {
            if (results[i].size() != k || gt[i].size() < k)
            {
                throw std::invalid_argument("each result row must contain exactly k IDs");
            }
            std::unordered_set<int> truth(gt[i].begin(), gt[i].begin() + static_cast<long>(k));
            std::size_t hits = 0;
            for (const auto& id : results[i])
            {
                if (truth.contains(static_cast<int>(id)))
                {
                    ++hits;
                }
            }
            //std::cout << static_cast<double>(hits) / static_cast<double>(k) << std::endl;
            recall_vec[i] = static_cast<double>(hits) / static_cast<double>(k);
            total += static_cast<double>(hits) / static_cast<double>(k);
        }
        const std::string rt = "recall(avg, P25, P50, 0): {} ";
        std::vector<float> return_result(4);
        return_result[0] = total / static_cast<double>(results.size());
        std::ranges::sort(recall_vec);
        return_result[1] = recall_vec[results.size() / 4 - 1];
        return_result[2] = recall_vec[results.size() / 2 - 1];
        return_result[3] = recall_vec[0];
        return return_result;
    }

    inline std::string quote(const std::string& value)
    {
        std::string escaped;
        escaped.reserve(value.size() + 2);
        escaped.push_back('"');
        for (const unsigned char character : value)
        {
            switch (character)
            {
            case '"': escaped += "\\\"";
                break;
            case '\\': escaped += "\\\\";
                break;
            case '\b': escaped += "\\b";
                break;
            case '\f': escaped += "\\f";
                break;
            case '\n': escaped += "\\n";
                break;
            case '\r': escaped += "\\r";
                break;
            case '\t': escaped += "\\t";
                break;
            default:
                if (character < 0x20)
                {
                    constexpr char hex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped.push_back(hex[character >> 4]);
                    escaped.push_back(hex[character & 0x0f]);
                }
                else
                {
                    escaped.push_back(static_cast<char>(character));
                }
            }
        }
        escaped.push_back('"');
        return escaped;
    }

    inline std::size_t parse_size(const std::string_view value, const std::string_view option)
    {
        std::size_t result = 0;
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
        if (error != std::errc{} || end != value.data() + value.size())
        {
            throw std::invalid_argument("invalid unsigned integer for --" + std::string(option));
        }
        return result;
    }

    inline void write_json(const std::string& path, const std::vector<std::pair<std::string, std::string>>& fields)
    {
        std::ofstream out(path);
        if (!out)
        {
            throw std::runtime_error("failed to write " + path);
        }
        out << "{\n";
        for (std::size_t i = 0; i < fields.size(); ++i)
        {
            out << "  \"" << fields[i].first << "\": " << fields[i].second;
            out << (i + 1 == fields.size() ? "\n" : ",\n");
        }
        out << "}\n";
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
            config.vector_dim = 1024;
            config.n_base_vectors = 1000000;
            config.n_query_vectors = 1000;
            config.n_learn_vectors = 100000;
        }
    }

    inline BenchmarkConfig parse_argv(int argc, char** argv)
    {
        std::unordered_map<std::string, std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc / 2));

        for (int i = 1; i < argc; i += 2)
        {
            const std::string_view option = argv[i];
            if (!option.starts_with("--"))
            {
                throw std::invalid_argument(
                    "expected an option beginning with --, got: "
                    + std::string(option));
            }
            if (i + 1 >= argc)
            {
                throw std::invalid_argument(
                    "missing value for option: " + std::string(option));
            }
            const std::string key(option.substr(2));
            const std::string value(argv[i + 1]);
            if (!arguments.emplace(key, value).second)
            {
                throw std::invalid_argument(
                    "duplicate option: --" + key);
            }
        }

        BenchmarkConfig config;
        const auto dataset = arguments.find("dataset");
        if (dataset == arguments.end())
        {
            throw std::invalid_argument("missing option: --dataset");
        }
        const auto dataset_type = parse_dataset_type(dataset->second);
        config.dataset_type = dataset_type;
        make_benchmark_with_default_params(config);

        for (const auto& [key, value] : arguments)
        {
            if (key == "dataset") {config.dataset = value;}
            else if (key == "engine") { config.engine_name = value; }
            else if (key == "mode") { config.engine_mode = value; }
            else if (key == "n_threads") { config.n_threads = parse_size(value, key); }
            else if (key == "path") { config.dataset_path = value; }
            else if (key == "dim") { config.vector_dim = parse_size(value, key); }
            else if (key == "n_base") { config.n_base_vectors = parse_size(value, key); }
            else if (key == "n_learn") { config.n_learn_vectors = parse_size(value, key); }
            else if (key == "n_query") { config.n_query_vectors = parse_size(value, key); }
            else if (key == "k") { config.k = parse_size(value, key); }
            else if (key == "kgt") { config.k_in_gt = parse_size(value, key); }
            else if (key == "result") { config.result = value; }
            else if (key == "reuse_index") { config.reuse_index = value == "1" ? true : false; }
            else if (key == "hs_page_size") { config.hs_page_size = parse_size(value, key); }
            else if (key == "hs_file_size") { config.hs_file_size = parse_size(value, key); }
            else if (key == "hs_n_ppage") { config.hs_n_pages_to_probe = parse_size(value, key); }
            else if (key == "hs_buf_size") { config.hs_local_buffer_size = parse_size(value, key); }
            else if (key == "hs_morsel") { config.hs_morsel_step = parse_size(value, key); }
            else if (key == "hs_beta") { config.hs_beta = std::stof(value); }
            else { throw std::invalid_argument("unknown option: --" + key); }
        }

        switch (dataset_type)
        {
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
            config.base_vector_path = (std::filesystem::path(config.dataset_path) / "base.fvecs").string();
            config.query_vector_path = (std::filesystem::path(config.dataset_path) / "query.fvecs").string();
            config.gt_vector_path = (std::filesystem::path(config.dataset_path) / "groundtruth.ivecs").string();
            config.learn_vector_path = config.dataset_path + "learn.fvecs";
            break;
        }
        if (config.engine_name.empty())
        {
            throw std::invalid_argument("missing option: --engine");
        }
        if (config.dataset_path.empty())
        {
            throw std::invalid_argument("missing option: --path");
        }
        if (config.n_threads == 0 || config.vector_dim == 0 || config.n_base_vectors == 0 ||
            config.n_query_vectors == 0 || config.k == 0 || config.k > config.k_in_gt ||
            config.k > config.n_base_vectors)
        {
            throw std::invalid_argument("invalid zero/count/k benchmark configuration");
        }

        return config;
    }

    std::uintmax_t directory_size(const std::filesystem::path& path)
    {
        std::uintmax_t size = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
        {
            if (entry.is_regular_file())
            {
                size += entry.file_size();
            }
        }
        return size;
    }

    void run(const BenchmarkConfig& config)
    {
        std::vector<std::vector<float>> vectors;
        double read_ms = 0.0;
        {
            ScopedTimer timer(read_ms);
            vectors = read_file_as_vectors(
                config.base_vector_path,
                config.n_base_vectors,
                config.vector_dim);
        }
        std::vector<std::size_t> ids(config.n_base_vectors);
        std::iota(ids.begin(), ids.end(), 0);

        std::string metrics_path = std::format("{}/{}-{}-{}-{}-{}.json", config.result, config.engine_name, config.engine_mode, config.hs_beta, config.hs_n_pages_to_probe, config.dataset);
        std::string index_path = std::format("{}/{}.{}", config.result, config.dataset, config.engine_name);

        double build_ms = 0.0;
        if (!config.reuse_index)
        {
            std::filesystem::remove_all(index_path);
            {
                ScopedTimer timer(build_ms);
                auto database = hyperspheres::create_database<float>(
                    index_path,
                    config.vector_dim,
                    hyperspheres::SupportedDistanceFunctions::SquaredL2Distance,
                    config.hs_page_size,
                    config.hs_file_size,
                    config.n_threads,
                    config.hs_local_buffer_size);
                hyperspheres::batch_prefill_parallel(database, vectors, ids);
            }
            vectors.clear();
            vectors.shrink_to_fit();
            ids.clear();
            ids.shrink_to_fit();
        }

        decltype(hyperspheres::load_database<float>(index_path)) database;
        double load_ms = 0.0;
        {
            ScopedTimer timer(load_ms);
            database = hyperspheres::load_database<float>(
                index_path,
                config.n_threads,
                config.hs_local_buffer_size);
        }

        auto queries = read_file_as_flat_array(
            config.query_vector_path,
            config.n_query_vectors,
            config.vector_dim);
        std::vector<std::vector<std::size_t>> results(config.n_query_vectors);
        std::vector<hyperspheres::IdTaggedDistance<float>> raw_results;
        bool aggressive = config.engine_mode == "aggressive";
        double search_ms = 0.0;
        {
            ScopedTimer timer(search_ms);
            for (std::size_t query = 0; query < config.n_query_vectors; ++query)
            {
                const auto nearest = hyperspheres::find_k_nearest(
                    database,
                    std::span<float>(queries.data() + query * config.vector_dim, config.vector_dim),
                    config.hs_beta,
                    aggressive,
                    config.k,
                    config.hs_n_pages_to_probe,
                    config.hs_morsel_step);
                auto& row = results[query];
                row.reserve(nearest.size());
                for (const auto& [distance, id] : nearest)
                {
                    static_cast<void>(distance);
                    row.push_back(id);
                }
            }
        }
        std::size_t sum = database->thread_pool->context->n_pruned.load();
        const auto n_db_pages = database->cached_max_dists.size();
        std::cout << static_cast<double>(config.n_query_vectors) << std::endl << static_cast<double>(database->
            cached_max_dists.size()) << std::endl;
        std::cout << static_cast<double>(sum) << std::endl;

        const auto ground_truth = read_gt_vectors(
            config.gt_vector_path,
            config.n_query_vectors,
            config.k_in_gt);
        const auto index_bytes = directory_size(index_path);
        database.reset();
        const auto recallsum = recall_summarize(results, ground_truth, config.k);
        write_json(metrics_path, {
                       {"engine", quote("hyperspheres")},
                       {"mode", quote(config.engine_mode)},
                       {"count", std::to_string(config.n_base_vectors)},
                       {"beta", std::to_string(config.hs_beta)},
                       {"dim", std::to_string(config.vector_dim)},
                       {"query_count", std::to_string(config.n_query_vectors)},
                       {"k", std::to_string(config.k)},
                       {"threads", std::to_string(config.n_threads)},
                       {"query_concurrency", "1"},
                       {"distance", quote("squared_l2")},
                       {"read_boundary", quote("read all base vectors into application memory")},
                       {"build_boundary", quote("create database and parallel batch prefill")},
                       {"read_ms", std::to_string(read_ms)},
                       {"build_ms", std::to_string(build_ms)},
                       {"load_ms", std::to_string(load_ms)},
                       {"search_ms", std::to_string(search_ms)},
                       {"avg_query_ms", std::to_string(search_ms / config.n_query_vectors)},
                       {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
                       {"recall_p25_at_k", std::to_string(recallsum[1])},
                       {"recall_p50_at_k", std::to_string(recallsum[2])},
                       {"recall_min", std::to_string(recallsum[3])},
                       {"total_pruned", std::to_string(sum)},
                       {"n_database_pages", std::to_string(n_db_pages)},
                       {
                           "pruning_rate",
                           std::to_string(
                               static_cast<double>(sum) / static_cast<double>(config.n_query_vectors) / static_cast<
                                   double>(config.hs_n_pages_to_probe == 0 ? n_db_pages : config.hs_n_pages_to_probe))
                       },
                       {"index_bytes", std::to_string(index_bytes)},
                       {"page_size", std::to_string(config.hs_page_size)},
                       {"file_size", std::to_string(config.hs_file_size)},
                       {"pages_to_probe", std::to_string(config.hs_n_pages_to_probe)},
                       {"morsel_step", std::to_string(config.hs_morsel_step)}
                   });
    }

    bool customized_prune_check_func(float dist_to_anchor, float in_page_max, float worst, float beta)
    {
        return dist_to_anchor > in_page_max && dist_to_anchor > worst;
    }
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config = parse_argv(argc, argv);
    run(config);
    return 0;
}
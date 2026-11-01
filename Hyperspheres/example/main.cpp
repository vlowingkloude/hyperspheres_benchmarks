#include <filesystem>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <queue>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <bit>

import hyperspheres;

namespace {

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

    std::vector<std::vector<int>> read_gt_vectors(const std::string& path, std::size_t rows, std::size_t dim) {
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

    double recall(const std::vector<std::vector<std::size_t>>& results,
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
            std::unordered_set<std::size_t> result (results[i].begin(), results[i].end());
            for (const auto& id : result) {
                if (truth.contains(static_cast<int>(id))) {
                    ++hits;
                }
            }
            total += static_cast<double>(hits) / static_cast<double>(k);
        }
        return total / static_cast<double>(results.size());
    }

    std::vector<std::vector<float>> read_file_as_vectors(const std::string& path, std::size_t rows, std::size_t dim) {
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
}

bool customized_prune_check_func(float dist_to_anchor, float in_page_max, float worst, float beta) {
    return dist_to_anchor > in_page_max && dist_to_anchor > worst;
}

int main() {
    std::string path = "testdb";
    std::string vector_path = "./dim_1024";

    constexpr std::size_t n_embeddings = 1000000;
    constexpr std::size_t n_dim = 1024;
    constexpr std::size_t n_queries = 1000;
    constexpr std::size_t k = 100;

    double build_ms = 0.0;
    {
        // prefill
        ScopedTimer timer(build_ms);
        std::filesystem::remove_all(path);
        std::vector<std::vector<float>> vectors = read_file_as_vectors(vector_path + "/base.fvecs", n_embeddings, n_dim);
        std::vector<std::size_t> ids(n_embeddings);
        std::iota(ids.begin(), ids.end(), 0);
        auto db = hyperspheres::create_database<float>(path, n_dim,
            hyperspheres::SupportedDistanceFunctions::SquaredL2Distance);
        hyperspheres::batch_prefill_parallel(db, vectors, ids);
    }
    std::cout << build_ms / 1000 << std::endl;

    double query_ms = 0.0;
    {
        //
        ScopedTimer timer(query_ms);
        auto db = hyperspheres::load_database<float>(path);
        auto queries = read_file_as_flat_array(vector_path + "/query.fvecs", n_queries, n_dim);
        std::vector<std::vector<std::size_t>> results(n_queries);
        for (std::size_t query = 0; query < n_queries; ++query) {
            const auto nearest = hyperspheres::find_k_nearest(
                db,
                std::span<float>(queries.data() + query * n_dim, n_dim), 0.7, true,
                k, 0,1, customized_prune_check_func);
            auto& row = results[query];
            row.reserve(nearest.size());
            for (const auto& [distance, id] : nearest) {
                static_cast<void>(distance);
                row.push_back(id);
            }
        }
        const auto ground_truth = read_gt_vectors(vector_path + "/groundtruth.ivecs", n_queries, k);
        auto rec = recall(results, ground_truth, k);
        std::cout << rec << std::endl;
    }
    std::cout << query_ms / n_queries << std::endl;

    return 0;
}
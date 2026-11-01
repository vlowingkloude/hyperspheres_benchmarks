#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/index_io.h>
#include <omp.h>

#include "benchmark_config.h"
#include "fvecs_stream.h"

namespace faiss_benchmark {

namespace {

void add_in_batches(
        faiss::Index& index,
        const BenchmarkConfig& config,
        const std::size_t batch_size) {
    FvecsBatchReader reader(config.base_vector_path, config.vector_dim);
    std::size_t added = 0;
    while (added < config.n_base_vectors) {
        const std::size_t rows = std::min(batch_size, config.n_base_vectors - added);
        const auto batch = reader.read(rows);
        index.add(static_cast<faiss::idx_t>(rows), batch.data());
        added += rows;
    }
}

} // namespace

void run(const BenchmarkConfig& config) {
    omp_set_num_threads(static_cast<int>(config.n_threads));
    faiss::IndexFlatL2 quantizer(config.vector_dim);
    std::unique_ptr<faiss::Index> index;

    double train_ms = 0.0;
    if (config.engine_mode == "flat") {
        index = std::make_unique<faiss::IndexFlatL2>(config.vector_dim);
    } else if (config.engine_mode == "hnsw") {
        auto hnsw = std::make_unique<faiss::IndexHNSWFlat>(
            config.vector_dim,
            static_cast<int>(config.faiss_hnsw_m),
            faiss::METRIC_L2);
        hnsw->hnsw.efConstruction = static_cast<int>(config.faiss_hnsw_efconstruction);
        hnsw->hnsw.efSearch = static_cast<int>(config.faiss_hnsw_efsearch);
        index = std::move(hnsw);
    } else if (config.engine_mode == "lsh") {
        throw std::invalid_argument(
            "FAISS LSH ranks binary codes by Hamming distance and is not comparable to squared L2");
    } else if (config.engine_mode == "ivf") {
        auto ivf = std::make_unique<faiss::IndexIVFFlat>(
            &quantizer,
            config.vector_dim,
            config.faiss_ivfflat_nlists,
            faiss::METRIC_L2);
        ivf->nprobe = config.faiss_ivfflat_nprobe;
        index = std::move(ivf);

        if (!config.reuse_index) {
            auto learn = read_file_as_flat_array(
            config.learn_vector_path,
            config.n_learn_vectors,
            config.vector_dim);
            {
                ScopedTimer timer(train_ms);
                index->train(static_cast<faiss::idx_t>(config.n_learn_vectors), learn.data());
            }
        }
    } else {
        throw std::invalid_argument("unknown FAISS mode: " + config.engine_mode);
    }

    double build_ms = 0.0;
    double save_ms = 0.0;
    if (!config.reuse_index) {
        {
            ScopedTimer timer(build_ms);
            add_in_batches(
                *index,
                config,
                config.faiss_ivfflat_ingest_batch_size);
        }
        {
            ScopedTimer timer(save_ms);
            faiss::write_index(index.get(), config.index_path.c_str());
        }
        index.reset();
    }

    double load_ms = 0.0;
    {
        ScopedTimer timer(load_ms);
        index.reset(faiss::read_index(config.index_path.c_str()));
    }
    if (config.engine_mode == "hnsw") {
        static_cast<faiss::IndexHNSWFlat*>(index.get())->hnsw.efSearch =
            static_cast<int>(config.faiss_hnsw_efsearch);
    } else if (config.engine_mode == "ivf") {
        static_cast<faiss::IndexIVFFlat*>(index.get())->nprobe =
            config.faiss_ivfflat_nprobe;
    }

    if (!config.reuse_index && config.engine_mode == "ivf") {
        write_json(config.metrics_path, {
        {"engine", quote("faiss")},
        {"mode", quote(config.engine_mode)},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"threads", std::to_string(config.n_threads)},
        {"query_concurrency", "1"},
        {"distance", quote("squared_l2")},
        {"build_boundary", quote("stream base vectors and add to index")},
        {"train_ms", std::to_string(train_ms)},
        {"build_ms", std::to_string(build_ms)},
        {"save_ms", std::to_string(save_ms)},
        {"load_ms", std::to_string(load_ms)}
    });
        return;
    }

    const auto queries = read_file_as_flat_array(
        config.query_vector_path,
        config.n_query_vectors,
        config.vector_dim);
    std::vector<float> distances(config.n_query_vectors * config.k);
    std::vector<faiss::idx_t> labels(config.n_query_vectors * config.k);

    double search_ms = 0.0;
    {
        ScopedTimer timer(search_ms);
	index->search(1000, queries.data(), static_cast<faiss::idx_t>(config.k), distances.data(), labels.data());
        /*for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
            index->search(
                1,
                queries.data() + query * config.vector_dim,
                static_cast<faiss::idx_t>(config.k),
                distances.data() + query * config.k,
                labels.data() + query * config.k);
        }*/
    }

    std::vector<std::vector<std::size_t>> results(config.n_query_vectors);
    for (std::size_t query = 0; query < config.n_query_vectors; ++query) {
        auto& row = results[query];
        row.reserve(config.k);
        for (std::size_t rank = 0; rank < config.k; ++rank) {
            const auto label = labels[query * config.k + rank];
            if (label < 0) {
                throw std::runtime_error("FAISS returned fewer than k neighbors");
            }
            row.push_back(static_cast<std::size_t>(label));
        }
    }
    const auto ground_truth = read_gt_vectors(
        config.gt_vector_path,
        config.n_query_vectors,
        config.k_in_gt);
    const auto base = read_file_as_flat_array(
        config.base_vector_path,
        config.n_base_vectors,
        config.vector_dim);
    const auto average_distance = avg_squared_l2_distance(
        std::span<const float>(queries.data(), queries.size()),
        std::span<const float>(base.data(), base.size()),
        config.vector_dim,
        results,
        ground_truth,
        config.k);

    write_json(config.metrics_path, {
        {"engine", quote("faiss")},
        {"mode", quote(config.engine_mode)},
        {"count", std::to_string(config.n_base_vectors)},
        {"dim", std::to_string(config.vector_dim)},
        {"query_count", std::to_string(config.n_query_vectors)},
        {"k", std::to_string(config.k)},
        {"threads", std::to_string(config.n_threads)},
        {"query_concurrency", "1"},
        {"distance", quote("squared_l2")},
        {"build_boundary", quote("stream base vectors and add to index")},
        {"train_ms", std::to_string(train_ms)},
        {"build_ms", std::to_string(build_ms)},
        {"save_ms", std::to_string(save_ms)},
        {"load_ms", std::to_string(load_ms)},
        {"search_ms", std::to_string(search_ms)},
        {"avg_query_ms", std::to_string(search_ms / config.n_query_vectors)},
        {"recall_at_k", std::to_string(recall(results, ground_truth, config.k))},
    {"avg_squared_l2_gt", std::to_string(average_distance.second)},
    {"avg_squared_l2", std::to_string(average_distance.first)},
        {"index_bytes", std::to_string(std::filesystem::file_size(config.index_path))},
        {"hnsw_m", std::to_string(config.faiss_hnsw_m)},
        {"hnsw_ef_construction", std::to_string(config.faiss_hnsw_efconstruction)},
        {"hnsw_ef_search", std::to_string(config.faiss_hnsw_efsearch)},
        {"ivf_nlist", std::to_string(config.faiss_ivfflat_nlists)},
        {"ivf_nprobe", std::to_string(config.faiss_ivfflat_nprobe)}
    });
}

} // namespace faiss_benchmark

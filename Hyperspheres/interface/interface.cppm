module;

#include <span>
#include <vector>
#include <memory>
#include <cmath>
#include <thread>
#include <cstring>
#include <algorithm>
#include <unordered_set>

export module hyperspheres;

import definitions;
import distance;
import storage;
import execution;
import database;

export namespace hyperspheres {

    template <typename ElemT, typename IdT = std::size_t>
        requires std::three_way_comparable<ElemT> && std::three_way_comparable<IdT>
    using IdTaggedDistance = structures::IdTaggedDistance<ElemT, IdT>;

    using DatabaseConfig = structures::DatabaseConfig;

    using SupportedDatatype = supported::SupportedDataType;
    using SupportedDistanceFunctions = supported::SupportedDistanceFunctions;
    using PruningMode = supported::PruningMode;

    template <typename T>
    std::unique_ptr<database::Database<T>> create_database(
        const std::string& path,
        const std::size_t dim,
        const supported::SupportedDistanceFunctions distance_function,
        const std::size_t page_size = defaults::default_page_size,
        const std::size_t file_size = defaults::default_single_file_size,
        const std::size_t n_threads = std::thread::hardware_concurrency(),
        const std::size_t local_distance_buffers_size = defaults::default_thread_local_buffer_size
        ) {
        supported::SupportedDataType dtype;
        if constexpr (std::is_same_v<T, float>) {
            dtype = supported::SupportedDataType::Float;
        }
        return database::create_database<T>(path, page_size, file_size, dtype, dim, distance_function, n_threads, local_distance_buffers_size);
    }

    template <typename T>
    std::unique_ptr<database::Database<T>> load_database(
        const std::string& path,
        const std::size_t n_threads = std::thread::hardware_concurrency(),
        const std::size_t local_distance_buffers_size = defaults::default_thread_local_buffer_size
        ) {
        return database::load_database<T>(path, n_threads, local_distance_buffers_size);
    }

    template <typename T>
    void batch_prefill(std::unique_ptr<database::Database<T>> &database,
        std::vector<std::vector<T>>& new_vectors,
        std::vector<std::size_t>& ids,
        const float in_page_vector_density = 1.0
        ) {
        execution::storage::batch_prefill_single_threaded(database->db_storage, new_vectors, ids,
            database->thread_pool->context->dist_func, in_page_vector_density);
        execution::storage::update_anchor_cache_single_threaded(database->db_storage, database->cached_vectors);
        execution::storage::update_cached_max_dist_single_threaded(database->db_storage, database->cached_max_dists);
    }

    template <typename T>
    void batch_prefill_parallel(std::unique_ptr<database::Database<T>> &database,
        std::vector<std::vector<T>>& new_vectors,
        std::vector<std::size_t>& ids,
        const float in_page_vector_density = 1.0
        ) {
        const auto context = database->thread_pool->context;
        context->task = hyperspheres::execution::parallel::ParallelTask::BatchPrefill;
        const std::size_t max_num_vectors_in_page = std::floor(database->db_storage.max_num_vectors_per_page * in_page_vector_density);
        for (std::size_t i = 1; i <= ids.size() / (max_num_vectors_in_page * database->db_storage.file_size / database->db_storage.page_size); ++i) {
            execution::storage::create_storage_file(database->db_storage);
        }
        context->scalar_input1 = max_num_vectors_in_page;
        database->internal_buffer.resize(max_num_vectors_in_page * database->thread_pool->n_threads);
        std::size_t page_idx {0};
        std::unordered_set<std::size_t> vector_indices;
        vector_indices.reserve(max_num_vectors_in_page);
        auto morsel_step = new_vectors.size() / database->thread_pool->n_threads / 2;
        if (morsel_step == 0) {
            morsel_step = 1;
        }
        context->morsel_step = morsel_step;
        while (true) {
            if (new_vectors.empty()) {
                execution::storage::update_anchor_cache_single_threaded(database->db_storage, database->cached_vectors);
                execution::storage::update_cached_max_dist_single_threaded(database->db_storage, database->cached_max_dists);
                return;
            }
            context->shared_data.store(std::numeric_limits<T>::infinity());
            context->generic_input4 = std::bit_cast<void *, std::vector<std::vector<T>>*>(&new_vectors);
            context->next_morsel.store(0);
            context->max_morsel = new_vectors.size();
            hyperspheres::execution::parallel::submit_and_block(database->thread_pool);
            std::size_t temp_offset = 0;
            for (std::size_t ti = 0; ti < database->thread_pool->n_threads; ti++) {
                temp_offset += database->n_outputs[ti];
            }
            database->internal_buffer.resize(temp_offset);
            temp_offset = 0;
            for (std::size_t ti = 0; ti < database->thread_pool->n_threads; ti++) {
                const auto n_out = database->n_outputs[ti];
                std::memcpy(
                    database->internal_buffer.data() + temp_offset,
                    database->local_distance_buffers.data() + static_cast<long>(ti * database->local_distance_buffers_size), n_out * sizeof(structures::IdTaggedDistance<float>)
                    );
                temp_offset += n_out;
            }
            std::size_t n_vectors = max_num_vectors_in_page;
            if (temp_offset < n_vectors) {
                n_vectors = temp_offset;
            }
            std::ranges::sort(database->internal_buffer);
            for (std::size_t ik = 0; ik < n_vectors; ik++) {
                const auto index = database->internal_buffer[ik].second;
                const auto id = ids[index];
                execution::storage::append_new_vector(database->db_storage, id, std::span<T>{new_vectors[index].data(), new_vectors[index].size()}, page_idx);
                vector_indices.emplace(index);
            }
            execution::storage::update_in_page_max_dist(database->db_storage, page_idx, &(database->internal_buffer[n_vectors-1].first), sizeof(T));
            std::size_t counter {0};
            std::erase_if(new_vectors, [&counter, &vector_indices](const auto&) {return vector_indices.contains(counter++);});
            counter = 0;
            std::erase_if(ids, [&counter, &vector_indices](const auto&) {return vector_indices.contains(counter++);});
            vector_indices.clear();
            page_idx++;
        }
    }

    template <typename T>
    void insert_a_vector(std::unique_ptr<database::Database<T>> &database,
        const std::size_t new_id,
        std::span<T> new_vector,
        const float in_page_vector_density = 1.0
        ) {
        execution::storage::insert_a_vector(
            database->db_storage, database->cached_vectors, database->cached_max_dists,
            database->dist_to_anchor_buffer, new_id, new_vector,
            database->thread_pool->context->dist_func, in_page_vector_density);
    }

    template <typename T>
    std::vector<IdTaggedDistance<T>> find_k_nearest(std::unique_ptr<database::Database<T>> &db,
        std::span<T> query, const float beta, const bool aggressive,
        const std::size_t k, const std::size_t n_pages_to_probe = 0, const std::size_t step = 1,
        bool (*customized_prune_check_func)(T, T, T, float) = nullptr) {
        std::vector<IdTaggedDistance<T>> result;
        result.reserve(k);
        const auto context = db->thread_pool->context;
        context->pruning_beta = beta;
        if (aggressive) {
            if (beta == 1.0) {
                context->prune_check_func = [](T dist_to_anchor, T in_page_max, T worst, float beta)  {return dist_to_anchor - in_page_max > worst;};
            } else {
                context->prune_check_func = [](T dist_to_anchor, T in_page_max, T worst, float beta) {return dist_to_anchor > worst && dist_to_anchor > in_page_max && dist_to_anchor - in_page_max * beta > worst; };
            }
        } else {
            context->prune_check_func = [](T dist_to_anchor, T in_page_max, T worst, float beta) {
                return (dist_to_anchor > in_page_max) && ((dist_to_anchor + in_page_max > worst)) &&
                    ( ((dist_to_anchor + in_page_max - worst) * (dist_to_anchor + in_page_max - worst)) >  4 * dist_to_anchor * in_page_max);
            };
        }
        if (customized_prune_check_func != nullptr) {
            context->prune_check_func = customized_prune_check_func;
        }
        context->vector_input2 = query;
        context->vector_input3 = db->cached_vectors;
        if (db->cached_max_dists.size() > db->thread_pool->n_threads * 2) [[likely]] {
            db->dist_to_anchor_buffer.resize(db->cached_max_dists.size());
            const auto n_fill_avg = db->cached_max_dists.size() / db->thread_pool->n_threads;
            const auto n_left = db->cached_max_dists.size() % db->thread_pool->n_threads;
            context->max_morsel = db->cached_max_dists.size();
            context->morsel_step = n_fill_avg;
            context->scalar_input1 = n_left;
            context->vector_input1 = db->cached_max_dists;
            context->vector_output_buffer4 = db->dist_to_anchor_buffer;
            context->task = hyperspheres::execution::parallel::ParallelTask::FillDistToAnchor;
            hyperspheres::execution::parallel::submit_and_block(db->thread_pool);
            std::ranges::sort(db->dist_to_anchor_buffer);
        } else {
            hyperspheres::execution::storage::fill_dist_to_anchor_buffer_single_threaded<T>(db->db_storage, db->cached_vectors,
            db->cached_max_dists, db->dist_to_anchor_buffer, query, db->thread_pool->context->dist_func, true);
        }
        context->next_morsel.store(0);
        if (n_pages_to_probe == 0 || n_pages_to_probe > db->dist_to_anchor_buffer.size()) {
            context->max_morsel = db->dist_to_anchor_buffer.size();
        } else {
            context->max_morsel = n_pages_to_probe;
        }
        context->task = hyperspheres::execution::parallel::ParallelTask::FindKnn;
        context->scalar_input1 = k;
        context->morsel_step = step;
        context->vector_input1 = db->dist_to_anchor_buffer;
        context->shared_data.store(std::numeric_limits<T>::infinity());
        hyperspheres::execution::parallel::submit_and_block(db->thread_pool);
        std::size_t temp_offset = 0;
        db->internal_buffer.resize(db->thread_pool->n_threads * k);
        for (std::size_t ti = 0; ti < db->thread_pool->n_threads; ti++) {
            const auto n_out = db->n_outputs[ti];
            std::memcpy(
                db->internal_buffer.data() + temp_offset,
                db->local_distance_buffers.data() + static_cast<long>(ti * db->local_distance_buffers_size), n_out * sizeof(structures::IdTaggedDistance<T>)
                );
            temp_offset += n_out;
        }
        db->internal_buffer.resize(temp_offset);
        std::ranges::sort(db->internal_buffer);
        for (std::size_t ik = 0; ik < std::min(k, temp_offset); ik++) {
            result.push_back(db->internal_buffer[ik]);
        }
        return std::move(result);
    }

    template <typename T>
    void normalize(std::vector<T>& x) {
        distance::normalize(std::span{x});
    }

}
module;

#include <cstddef>
#include <string>
#include <format>
#include <filesystem>
#include <memory>

export module database;

import definitions;
import memorymap;
import storage;
import execution;
import distance;

export namespace hyperspheres::database {

    template <typename T>
    struct Database {
        hyperspheres::storage::Storage db_storage;
        hyperspheres::memorymap::MappedFile config_file;
        hyperspheres::structures::DatabaseConfig *db_config;
        hyperspheres::execution::parallel::ThreadPool<T> *thread_pool;
        // caches and runtime buffers
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> local_distance_buffers;
        std::size_t local_distance_buffers_size;
        std::vector<T> cached_vectors;
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> cached_max_dists;
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> dist_to_anchor_buffer;
        std::vector<std::size_t> n_outputs;
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> internal_buffer;

        Database() = delete;

        // load
        Database(const std::string& path, const std::size_t n_threads, const std::size_t buffers_size)
            : config_file(path + "/" + hyperspheres::defaults::config_filename, hyperspheres::memorymap::MmapMode::ReadWrite) {
            db_config = std::bit_cast<hyperspheres::structures::DatabaseConfig *, std::byte *>(config_file.addr);
            db_storage = hyperspheres::execution::storage::load_storage(path, db_config);
            local_distance_buffers = std::vector<hyperspheres::structures::IdTaggedDistance<T>>(n_threads * buffers_size);
            local_distance_buffers_size = buffers_size;
            n_outputs = std::vector<std::size_t>(n_threads);
            thread_pool = new hyperspheres::execution::parallel::ThreadPool<T>(db_storage, n_threads, buffers_size, local_distance_buffers, n_outputs);
            cached_vectors = std::vector<T>(db_storage.n_pages * db_storage.vector_dim);
            cached_max_dists = std::vector<hyperspheres::structures::IdTaggedDistance<T>>{};
            cached_max_dists.reserve(db_storage.n_pages);
            dist_to_anchor_buffer = std::vector<hyperspheres::structures::IdTaggedDistance<T>>{};
            dist_to_anchor_buffer.reserve(db_storage.n_pages);
            internal_buffer = std::vector<hyperspheres::structures::IdTaggedDistance<T>>{};
            internal_buffer.reserve(hyperspheres::defaults::default_thread_local_buffer_size);
            switch (db_config->distance_function) {
                case hyperspheres::supported::SupportedDistanceFunctions::SquaredL2Distance:
                    thread_pool->context->dist_func = hyperspheres::distance::squared_l2_distance;
                    break;
                default:
                    throw std::invalid_argument("Unknown distance function");
            }
        }

        ~Database() {
            db_config->n_files = db_storage.n_files;
            delete thread_pool;
        }
    };

    template <typename T>
    std::unique_ptr<Database<T>> load_database(const std::string& path, const std::size_t n_threads, const std::size_t local_distance_buffers_size) {
        auto db = std::make_unique<Database<T>>(path, n_threads, local_distance_buffers_size);
        db->thread_pool->context->thread_local_output_buffer1 = db->local_distance_buffers;
        db->thread_pool->context->vector_output_buffer2 = db->cached_vectors;
        db->thread_pool->context->vector_output_buffer3 = db->n_outputs;
        execution::storage::update_anchor_cache_single_threaded(db->db_storage, db->cached_vectors);
        execution::storage::update_cached_max_dist_single_threaded(db->db_storage, db->cached_max_dists);
        return db;
    }

    template <typename T>
    std::unique_ptr<Database<T>> create_database(const std::string& path, const std::size_t page_size, const std::size_t file_size,
            const hyperspheres::supported::SupportedDataType dtype, const std::size_t dim,
            const hyperspheres::supported::SupportedDistanceFunctions distance_function,
            const std::size_t n_threads, const std::size_t local_distance_buffers_size) {
        {
            const std::filesystem::path dbpath{path};
            std::filesystem::create_directory(dbpath);
            hyperspheres::execution::storage::create_storage(path, page_size, file_size, dtype, dim);
            hyperspheres::execution::storage::create_config_file(path, dtype, distance_function, page_size, file_size, dim);
        }
        auto db = std::make_unique<Database<T>>(path, n_threads, local_distance_buffers_size);
        db->thread_pool->context->thread_local_output_buffer1 = db->local_distance_buffers;
        db->thread_pool->context->vector_output_buffer2 = db->cached_vectors;
        db->thread_pool->context->vector_output_buffer3 = db->n_outputs;
        return db;
    }

}
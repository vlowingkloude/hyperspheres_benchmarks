module;

#include <algorithm>
#include <filesystem>
#include <string>
#include <cstring>
#include <format>
#include <atomic>
#include <vector>
#include <thread>
#include <cmath>
#include <unordered_set>
#include <bit>
#include <mdspan>

export module execution;

import definitions;
import storage;
import memorymap;

export namespace hyperspheres::execution::storage {

    void create_config_file(const std::string& name,
        const hyperspheres::supported::SupportedDataType dtype,
        const hyperspheres::supported::SupportedDistanceFunctions distance_function,
        const std::size_t page_size, const std::size_t file_size, const std::size_t dim) {
        const auto path = name + "/" + hyperspheres::defaults::config_filename;
        const auto config_file = memorymap::MappedFile(path, memorymap::MmapMode::Write, sizeof(hyperspheres::structures::DatabaseConfig));
        auto *config = std::bit_cast<hyperspheres::structures::DatabaseConfig *, std::byte *>(config_file.addr);
        config->n_files = 1;
        config->page_size = page_size;
        config->file_size = file_size;
        config->dim = dim;
        config->distance_function = distance_function;
        config->dtype = dtype;
    }

    void create_storage_file(hyperspheres::storage::Storage& db_storage) {
        const std::string path = std::format("{}/{}", db_storage.path, db_storage.n_files);
        {
            hyperspheres::storage::MappedDataFile new_file {path, db_storage.page_size, db_storage.file_size};
        }
        db_storage.files.emplace_back(path, db_storage.page_size);
        db_storage.n_files += 1;
        db_storage.n_pages += (db_storage.file_size / db_storage.page_size);
    }

    void create_storage(const std::string& name, const std::size_t page_size, const std::size_t file_size, const hyperspheres::supported::SupportedDataType dtype, const std::size_t dim) {
        const std::filesystem::path path {name};
        std::filesystem::create_directory(path);
        std::size_t element_size;
        switch (dtype) {
            case hyperspheres::supported::SupportedDataType::Float:
                element_size = sizeof(float);
                break;
            default:
                throw std::invalid_argument("Unsupported data type");
        }
        constexpr auto id_offset = hyperspheres::defaults::first_n_bytes_per_page;
        const auto max_num_vectors_per_page = (page_size - id_offset) / (sizeof(id_offset) + element_size * dim);
        const auto vector_offset = id_offset + sizeof(id_offset) * max_num_vectors_per_page;
        hyperspheres::storage::Storage db_storage {name, {}, 0, page_size, file_size, id_offset, vector_offset, 0, element_size, dim, max_num_vectors_per_page};
        create_storage_file(db_storage);
    }

    hyperspheres::storage::Storage load_storage(const std::string& name, hyperspheres::structures::DatabaseConfig *config) {
        std::size_t element_size;
        switch (config->dtype) {
            case hyperspheres::supported::SupportedDataType::Float:
                element_size = sizeof(float);
                break;
            default:
                throw std::invalid_argument("Unsupported data type");
        }
        constexpr auto id_offset = hyperspheres::defaults::first_n_bytes_per_page;
        const auto max_num_vectors_per_page = (config->page_size - id_offset) / (sizeof(id_offset) + element_size * config->dim);
        const auto vector_offset = id_offset + sizeof(id_offset) * max_num_vectors_per_page;
        const std::size_t n_pages = config->n_files * config->file_size / config->page_size;
        hyperspheres::storage::Storage db_storage {name, {}, config->n_files,
            config->page_size, config->file_size, id_offset, vector_offset, n_pages, element_size, config->dim, max_num_vectors_per_page};

        db_storage.files.reserve(config->n_files);
        for (std::size_t i = 0; i < config->n_files; i++) {
            db_storage.files.emplace_back(std::format("{}/{}", name, i), config->page_size);
        }

        return std::move(db_storage);
    }

    inline void place_new_vector_at(const hyperspheres::storage::Storage& db_storage, const std::size_t new_id,
        const void* const new_vector, const std::size_t element_size, const std::size_t dim,
        const std::size_t file_index, const std::size_t page_index, const std::size_t vector_index) {
        std::memcpy(db_storage.files[file_index].pages[page_index].page_addr + db_storage.vector_offset + vector_index * dim * element_size,
            new_vector, static_cast<long>(element_size * dim));
        std::memcpy(db_storage.files[file_index].pages[page_index].page_addr + db_storage.id_offset + vector_index * sizeof(std::size_t), &new_id, sizeof(std::size_t));
    }

    inline std::size_t calculate_file_index(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx) {
        return page_idx / (db_storage.file_size / db_storage.page_size);
    }

    inline std::size_t calculate_page_index(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx) {
        return page_idx % (db_storage.file_size / db_storage.page_size);
    }

    inline std::size_t& get_n_vectors_of(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index) {
        return *(db_storage.files[file_index].pages[page_index].n_vectors);
    }

    inline std::size_t& get_n_vectors_of(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        return get_n_vectors_of(db_storage, file_index, page_index);
    }

    inline void update_number_of_vectors_at(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, const long by = 1) {
        auto& n = get_n_vectors_of(db_storage, file_index, page_index);
        n += by;
    }

    inline void update_in_page_max_dist(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, const void * const data, const std::size_t element_size) {
        std::memset(db_storage.files[file_index].pages[page_index].max_dist, 0, hyperspheres::defaults::max_n_bytes_of_native_type);
        std::memcpy(db_storage.files[file_index].pages[page_index].max_dist, data, element_size);
    }

    inline void update_in_page_max_dist(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx, const void * const data, const std::size_t element_size) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        update_in_page_max_dist(db_storage, file_index, page_index, data, element_size);
    }

    template <typename T>
    T get_max_dist_at(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index) {
        return *(std::bit_cast<T *, std::byte *>(db_storage.files[file_index].pages[page_index].max_dist));
    }

    template <typename T>
    T get_max_dist_at(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        return get_max_dist_at<T>(db_storage, file_index, page_index);
    }

    template <typename T>
    void place_new_vector_at(const hyperspheres::storage::Storage& db_storage, const std::size_t new_id,
        const std::span<T>& new_vector, const std::size_t file_index, const std::size_t page_index, const std::size_t vector_index) {
        place_new_vector_at(db_storage, new_id, new_vector.data(), sizeof(T), new_vector.size(), file_index, page_index, vector_index);
    }

    template <typename T>
    void append_new_vector(const hyperspheres::storage::Storage& db_storage, const std::size_t new_id,
        const std::span<T>& new_vector, const std::size_t file_index, const std::size_t page_index) {
        const auto n = get_n_vectors_of(db_storage, file_index, page_index);
        place_new_vector_at<T>(db_storage, new_id, new_vector, file_index, page_index, n);
        update_number_of_vectors_at(db_storage, file_index, page_index);
    }

    template <typename T>
    void append_new_vector(const hyperspheres::storage::Storage& db_storage, const std::size_t new_id,
        const std::span<T>& new_vector, const std::size_t page_idx) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        append_new_vector<T>(db_storage, new_id, new_vector, file_index, page_index);
    }

    template <typename T>
    std::mdspan<T, std::dextents<std::size_t, 2>> get_array_view_of_page(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index) {
        if (db_storage.files.empty()) [[unlikely]] {
            return {};
        }
        if (file_index >= db_storage.files.size() || page_index >= db_storage.n_pages / db_storage.n_files) [[unlikely]] {
            return {};
        }
        const auto base_ptr = db_storage.files[file_index].pages[page_index].page_addr;
        const auto n_vectors = *(db_storage.files[file_index].pages[page_index].n_vectors);
        if (n_vectors == 0) {
            return {};
        }
        return std::mdspan<T, std::dextents<std::size_t, 2>>{std::bit_cast<T *, std::byte *>(base_ptr + db_storage.vector_offset), n_vectors, db_storage.vector_dim};
    }

    template <typename T>
    std::mdspan<T, std::dextents<std::size_t, 2>> get_array_view_of_page(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        return get_array_view_of_page<T>(db_storage, file_index, page_index);
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    std::size_t find_victim_single_threaded(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, T(*dist_func)(std::span<T>, std::span<T>)) {
        const auto max_dist = get_max_dist_at<T>(db_storage, file_index, page_index);
        const auto n_vectors = get_n_vectors_of(db_storage, file_index, page_index);
        auto view = get_array_view_of_page<T>(db_storage, file_index, page_index);
        const auto anchor = view[0];
        for (std::size_t i = 1; i < n_vectors; ++i) {
            if (max_dist == dist_func(view[i], anchor)) {
                return i;
            }
        }
        return n_vectors;
    }

    inline std::size_t get_vector_id(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, const std::size_t vector_index) {
        return *(std::bit_cast<std::size_t *, std::byte*>(db_storage.files[file_index].pages[page_index].page_addr + db_storage.id_offset) + vector_index);
    }

    inline std::size_t get_vector_id(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx, const std::size_t vector_index) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        return get_vector_id(db_storage, file_index, page_index, vector_index);
    }

    template <typename T>
    std::span<T> get_vector(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, const std::size_t vector_index) {
        const auto dim = db_storage.vector_dim;
        return std::span<T> {std::bit_cast<T *, std::byte *>(db_storage.files[file_index].pages[page_index].page_addr + db_storage.vector_offset + vector_index * dim * sizeof(T)), dim};
    }

    template <typename T>
    std::span<T> get_vector(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx, const std::size_t vector_index) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        return get_vector<T>(db_storage, file_index, page_index, vector_index);
    }

    template <typename T>
    std::vector<T> buffer_victim(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, const std::size_t victim_index) {
        std::vector<T> victim(db_storage.vector_dim);
        std::memcpy(victim.data(), db_storage.files[file_index].pages[page_index].page_addr + db_storage.vector_offset + victim_index * db_storage.vector_dim * sizeof(T), db_storage.vector_dim * sizeof(T));
        return std::move(victim);
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T calculate_new_max_dist(const hyperspheres::storage::Storage& db_storage, const std::size_t file_index, const std::size_t page_index, T(*dist_func)(std::span<T>, std::span<T>)) {
        const auto n_vectors = get_n_vectors_of(db_storage, file_index, page_index);
        auto view = get_array_view_of_page<T>(db_storage, file_index, page_index);
        const auto anchor = view[0];
        T max_dist {};
        for (std::size_t i = 1; i < n_vectors; ++i) {
            const auto dist = dist_func(view[i], anchor);
            if (max_dist < dist) {
                max_dist = dist;
            }
        }
        return max_dist;
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T calculate_new_max_dist(const hyperspheres::storage::Storage& db_storage, const std::size_t page_idx, T(*dist_func)(std::span<T>, std::span<T>)) {
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        return calculate_new_max_dist(db_storage, file_index, page_index, dist_func);
    }

    template <typename T>
    void update_anchor_cache_single_threaded(const hyperspheres::storage::Storage& db_storage, std::vector<T>& cached_vectors) {
        if (cached_vectors.size() < db_storage.n_pages * db_storage.vector_dim) {
            cached_vectors.resize(db_storage.n_pages * db_storage.vector_dim);
        }
        for (std::size_t i = 0; i < db_storage.n_pages; ++i) {
            if (get_n_vectors_of(db_storage, i) == 0) [[unlikely]] {
                continue;
            }
            std::memcpy(cached_vectors.data() + i * db_storage.vector_dim, get_vector<T>(db_storage, i, 0).data(), db_storage.vector_dim * sizeof(T));
        }
    }

    template <typename T>
    void update_cached_max_dist_single_threaded(const hyperspheres::storage::Storage& db_storage, std::vector<hyperspheres::structures::IdTaggedDistance<T>>& cached_max_dists) {
        if (cached_max_dists.capacity() < db_storage.n_pages) {
            cached_max_dists.reserve(db_storage.n_pages);
        }
        cached_max_dists.clear();
        for (std::size_t i = 0; i < db_storage.n_pages; ++i) {
            const auto n = get_n_vectors_of(db_storage, i);
            if (n > 0) {
                cached_max_dists.emplace_back(get_max_dist_at<T>(db_storage, i), i);
            }
        }
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void fill_dist_to_anchor_buffer_single_threaded(const hyperspheres::storage::Storage& db_storage,
        std::vector<T> cached_vectors,
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> cached_max_dists,
        std::vector<hyperspheres::structures::IdTaggedDistance<T>>& dist_to_anchor_buffer,
        std::span<T> query,
        T(*dist_func)(std::span<T>, std::span<T>), const bool sort) {
        if (dist_to_anchor_buffer.capacity() < cached_max_dists.size()) {
            dist_to_anchor_buffer.reserve(cached_max_dists.size());
        }
        dist_to_anchor_buffer.clear();
        for (const auto& [_, idx] : cached_max_dists) {
            dist_to_anchor_buffer.emplace_back(dist_func(query, std::span<T>{cached_vectors.data() + idx * db_storage.vector_dim, db_storage.vector_dim}), idx);
        }
        if (sort) {
            std::ranges::sort(dist_to_anchor_buffer);
        }
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void insert_a_vector(hyperspheres::storage::Storage& db_storage,
        std::vector<T> &cached_anchors,
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> &cached_max_dists,
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> &dist_to_anchor_buffer,
        const std::size_t new_id,
        std::span<T> new_vector,
        T(*dist_func)(std::span<T>, std::span<T>), const float in_page_vector_density) {
        // case 1: empty database -> copy new vector
        // case 2: best page has one vector only -> copy new vector, update max_dist
        // case 3: current_best_dist < max_dist of the best page, and best page is not full -> copy new vector
        // case 4: current_best_dist < max_dist of the best page, but best page is full ->
        //          find the victim which is the farthest vector, copy the victim into a buffer, copy new vector, update max_dist, insert victim
        // case 5: current_best_dist >= max_dist, and there is an empty page in current database -> copy the new vector
        // case 6: current_best_dist >= max_dist, but there is no empty page in current database, and the file is dense ->
        //          create a new db file, copy the new vector (basically the first page in new file)
        // case 7: current_best_dist >= max_dist, but there is no empty page in current database, the file is sparse, and the best page is not full ->
        //          copy the new vector, update max_dist
        // case 8: current_best_dist >= max_dist, but there is no empty page in current database, the file is sparse, but the best page is full ->
        //          create a new db file, copy the new vector (basically the first page in new file)

        if (dist_to_anchor_buffer.empty()) {
            // case 1
            append_new_vector(db_storage, new_id, new_vector, 0);
            update_anchor_cache_single_threaded(db_storage, cached_anchors);
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            return;
        }
        const auto page_idx = dist_to_anchor_buffer[0].second;
        const auto n_vectors = get_n_vectors_of(db_storage, page_idx);
        if (n_vectors == 1) {
            // case 2
            append_new_vector(db_storage, new_id, new_vector, page_idx);
            T new_max_dist = calculate_new_max_dist(db_storage, page_idx, dist_func);
            update_in_page_max_dist(db_storage, page_idx, &new_max_dist, sizeof(T));
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            return;
        }
        const T current_best_dist = dist_to_anchor_buffer[0].first;
        const T max_dist = get_max_dist_at<T>(db_storage, page_idx);
        const std::size_t max_num_vectors = std::floor(static_cast<float>(db_storage.max_num_vectors_per_page) * in_page_vector_density);
        if (current_best_dist < max_dist && n_vectors + 1 <= max_num_vectors) {
            // case 3
            append_new_vector(db_storage, new_id, new_vector, page_idx);
            return;
        }
        const auto file_index = calculate_file_index(db_storage, page_idx);
        const auto page_index = calculate_page_index(db_storage, page_idx);
        if (current_best_dist < max_dist && n_vectors + 1 > max_num_vectors) {
            // case 4
            const auto victim_index = find_victim_single_threaded(db_storage, file_index, page_index, dist_func);
            const auto victim_id = get_vector_id(db_storage, file_index, page_index, victim_index);
            auto victim = buffer_victim<T>(db_storage, file_index, page_index, victim_index);
            place_new_vector_at(db_storage, new_id, new_vector.data(), sizeof(T), db_storage.vector_dim, file_index, page_index, victim_index);
            const T new_max_dist = calculate_new_max_dist(db_storage, file_index, page_index, dist_func);
            update_in_page_max_dist(db_storage, file_index, page_index, &new_max_dist, sizeof(T));
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            fill_dist_to_anchor_buffer_single_threaded(db_storage, cached_anchors, cached_max_dists, dist_to_anchor_buffer, std::span<T>{victim}, dist_func, true);
            insert_a_vector(db_storage, cached_anchors, cached_max_dists, dist_to_anchor_buffer, victim_id, std::span<T>{victim}, dist_func, in_page_vector_density);
            return;
        }
        if (current_best_dist >= max_dist && cached_max_dists.size() < db_storage.n_pages) {
            // case 5
            // find an empty page
            std::vector<char> all_page_idx (db_storage.n_pages, 1);
            for (const auto [_, idx] : cached_max_dists) {
                all_page_idx[idx] = 0;
            }
            std::size_t empty_page_idx = std::ranges::distance(all_page_idx.begin(), std::ranges::find(all_page_idx, 1));
            append_new_vector(db_storage, new_id, new_vector, empty_page_idx);
            update_anchor_cache_single_threaded(db_storage, cached_anchors);
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            return;
        }
        if (current_best_dist >= max_dist && cached_max_dists.size() >= db_storage.n_pages) {
            // case 6
            create_storage_file(db_storage);
            append_new_vector(db_storage, new_id, new_vector, db_storage.files.size() - 1, 0);
            update_anchor_cache_single_threaded(db_storage, cached_anchors);
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            return;
        }
        if (current_best_dist >= max_dist && n_vectors + 1 < max_num_vectors) {
            // case 7
            append_new_vector(db_storage, new_id, new_vector, file_index, page_index);
            update_in_page_max_dist(db_storage, file_index, page_index, &current_best_dist, sizeof(T));
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            return;
        }
        if (current_best_dist >= max_dist && n_vectors + 1 >= max_num_vectors) {
            // case 8
            create_storage_file(db_storage);
            append_new_vector(db_storage, new_id, new_vector, db_storage.files.size() - 1, 0);
            update_anchor_cache_single_threaded(db_storage, cached_anchors);
            update_cached_max_dist_single_threaded(db_storage, cached_max_dists);
            return;
        }
        throw std::runtime_error("bug, this case is not handled");
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void batch_prefill_single_threaded(hyperspheres::storage::Storage& db_storage,
        std::vector<std::vector<T>>& new_vectors, std::vector<std::size_t>& ids,
        T(*dist_func)(std::span<T>, std::span<T>), const float in_page_vector_density) {
        // todo: better check if the database is empty
        const std::size_t max_num_vectors_per_page = std::floor(static_cast<float>(db_storage.max_num_vectors_per_page) * in_page_vector_density);

        // prepare database files
        // there is already one file when we create the database
        for (std::size_t i = 1; i <= ids.size() / (max_num_vectors_per_page * db_storage.file_size / db_storage.page_size); ++i) {
            create_storage_file(db_storage);
        }

        std::size_t page_idx {0};
        std::vector<hyperspheres::structures::IdTaggedDistance<T>> n_nearest;
        const std::size_t max_num_nearest = max_num_vectors_per_page - 1;
        n_nearest.reserve(max_num_nearest);
        std::unordered_set<std::size_t> vector_indices;
        vector_indices.reserve(max_num_vectors_per_page);


        while (true) {
            if (new_vectors.empty()) {
                return;
            }
            for (std::size_t vector_index = 1; vector_index < new_vectors.size(); ++vector_index) {
                const auto dist = dist_func(new_vectors[0], new_vectors[vector_index]);
                if (n_nearest.size() < max_num_nearest) {
                    n_nearest.emplace_back(dist, vector_index);
                    std::ranges::push_heap(n_nearest.begin(), n_nearest.end());
                } else if (n_nearest[0].first > dist) {
                    std::ranges::pop_heap(n_nearest.begin(), n_nearest.end());
                    n_nearest[max_num_nearest - 1] = {dist, vector_index};
                    std::ranges::push_heap(n_nearest.begin(), n_nearest.end());
                }
            }
            append_new_vector<T>(db_storage, ids[0], new_vectors[0], page_idx);
            vector_indices.emplace(0);
            T max_dist {};
            if (!n_nearest.empty()) {
                max_dist = n_nearest[0].first;
            }
            for (const auto [_, idx] : n_nearest) {
                append_new_vector<T>(db_storage, ids[idx], new_vectors[idx], page_idx);
                vector_indices.emplace(idx);
            }
            update_in_page_max_dist(db_storage, page_idx, &max_dist, sizeof(T));
            std::size_t counter {0};
            std::erase_if(new_vectors, [&counter, &vector_indices](const auto&) {return vector_indices.contains(counter++);});
            counter = 0;
            std::erase_if(ids, [&counter, &vector_indices](const auto&) {return vector_indices.contains(counter++);});
            vector_indices.clear();
            n_nearest.clear();
            page_idx++;
        }
    }

}

export namespace hyperspheres::execution::parallel {

    enum class ParallelTask : std::uint32_t {
        Nop, // just a placeholder
        FillDistToAnchor,
        FindKnn, // page level
        BatchPrefill,
        Exit, // exiting threads
    };

    template <typename T>
    struct ParallelContext {
        hyperspheres::storage::Storage& db_storage;
        std::atomic<std::size_t> next_morsel {};
        std::size_t morsel_step = 0;
        std::size_t max_morsel = 0;
        ParallelTask task = ParallelTask::Nop;
        volatile std::size_t scalar_input1 = 0; // usually k
        volatile std::size_t scalar_input2 = 0;
        volatile T scalar_input3 {};
        std::span<hyperspheres::structures::IdTaggedDistance<T>> vector_input1 {}; // usually the cache
        std::span<T> vector_input2 {}; // usually the query
        std::span<T> vector_input3 {}; // usually the cached vectors
        void * generic_input4 = nullptr; //
        std::span<hyperspheres::structures::IdTaggedDistance<T>> thread_local_output_buffer1 {}; // usually distances
        std::span<T> vector_output_buffer2 {}; // usually for copying / caching vectors
        std::span<std::size_t> vector_output_buffer3 {}; // n outputs of each threads
        std::span<hyperspheres::structures::IdTaggedDistance<T>> vector_output_buffer4 {}; // usually ID tagged distances with orders
        std::atomic<T> shared_data {}; // usually for distance
        std::atomic<std::size_t> shared_index {};
        std::atomic<std::size_t> n_pruned;
        std::atomic<bool> thread_control {false}; // usually to tell threads that they can stop
        T (*dist_func)(std::span<T>, std::span<T>) {};
        float pruning_beta {1};
        bool (*prune_check_func)(T, T, T, float) {};

        explicit ParallelContext(hyperspheres::storage::Storage& db_storage) : db_storage(db_storage) {}
    };

    template <typename T>
    void hyperspheres_worker(ParallelContext<T>* context, std::size_t thread_id, std::size_t buffer_size, std::atomic<std::size_t>* global_task_id, std::atomic<bool> *job_done);

    template <typename T>
    struct ThreadPool {
        std::size_t n_threads = 0;
        std::vector<std::jthread> threads;
        std::atomic<std::size_t> *global_task_id;
        std::vector<std::atomic<bool> *> thread_job_done;
        ParallelContext<T> *context;

        ThreadPool(hyperspheres::storage::Storage& db_storage, const std::size_t n, const std::size_t buffer_size, std::span<hyperspheres::structures::IdTaggedDistance<T>> thread_local_output_buffer1, std::span<std::size_t> vector_output_buffer3) : n_threads(n) {
            global_task_id = new std::atomic<std::size_t>(0);
            context = new ParallelContext<T>(db_storage);
            context->thread_local_output_buffer1 = thread_local_output_buffer1;
            context->vector_output_buffer3 = vector_output_buffer3;
            context->n_pruned.store(0);
            for (std::size_t i = 0; i < n_threads; i++) {
                auto job_done = new std::atomic<bool>(true);
                thread_job_done.emplace_back(job_done);
            }
            for (std::size_t i = 0; i < n_threads; i++) {
                threads.emplace_back(hyperspheres_worker<T>, context, i, buffer_size, global_task_id, thread_job_done[i]);
            }
        }

        ~ThreadPool() {
            context->task = hyperspheres::execution::parallel::ParallelTask::Exit;
            for (std::size_t i = 0; i < n_threads; i++) {
                thread_job_done[i]->store(false, std::memory_order_release);
            }
            global_task_id->fetch_add(1,std::memory_order_release);
            global_task_id->notify_all();
            for (std::size_t i = 0; i < n_threads; i++) {
                thread_job_done[i]->wait(false, std::memory_order_acquire);
            }
            delete global_task_id;
            delete context;
        }
    };
}

namespace hyperspheres::execution::parallel::workers {

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void find_victim_vector_index_worker(ParallelContext<T> *context, std::size_t thread_id) {
        const auto n_vectors = context->max_morsel;
        const auto step = context->morsel_step;
        const auto func = context->dist_func;
        const auto file_index = context->scalar_input1;
        const auto page_index = context->scalar_input2;
        const auto view = hyperspheres::execution::storage::get_array_view_of_page<T>(context->db_storage, file_index, page_index);
        const T max_dist = hyperspheres::execution::storage::get_max_dist_at<T>(context->db_storage, file_index, page_index);
        while (true) {
            const auto morsel = context->next_morsel.fetch_add(step);
            if (morsel >= n_vectors) {
                return;
            }
            const auto limit = std::min(n_vectors, step + morsel);
            for (std::size_t i = morsel; i < limit; ++i) {
                const auto dist = func(view[i], context->vector_input2);
                if (dist == max_dist) {
                    context->shared_index.store(i);
                    return;
                }
            }
            if (context->thread_control.load()) {
                return;
            }
        }
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void batch_prefill_worker(ParallelContext<T> *context, std::size_t thread_id, std::span<hyperspheres::structures::IdTaggedDistance<T>> heap, std::size_t &n_outputs) {
        const auto dim = context->db_storage.vector_dim;
        const auto max_morsel = context->max_morsel;
        const auto step = context->morsel_step;
        const auto max_num_vectors_in_page = context->scalar_input1;
        auto func = context->dist_func;

        while (true) {
            const auto morsel = context->next_morsel.fetch_add(step);
            auto &vectors = *std::bit_cast<std::vector<std::vector<T>> *, void *>(context->generic_input4);
            if (morsel >= max_morsel) {
                return;
            }
            const auto current_global_max = context->shared_data.load();
            const std::span<T> anchor = std::span{vectors[0].begin(), vectors[0].size()};
            for (std::size_t current = morsel; current < std::min(max_morsel, step + morsel); ++current) {
                const auto v = std::span{vectors[current].begin(), vectors[current].size()};
                const auto dist = func(v, anchor);
                if (dist > current_global_max) {
                    continue;
                }
                if (n_outputs < max_num_vectors_in_page) {
                    heap[n_outputs++] = {dist, current};
                    std::ranges::push_heap(heap.begin(), heap.begin() + n_outputs);
                } else if (dist < heap[0].first) {
                    const std::size_t vector_id = current;
                    std::ranges::pop_heap(heap.begin(), heap.begin() + n_outputs);
                    heap[max_num_vectors_in_page-1] = {dist, vector_id};
                    std::ranges::push_heap(heap.begin(), heap.begin() + n_outputs);
                }
            }
            if (heap[0].first < current_global_max && n_outputs == max_num_vectors_in_page) {
                auto up_to_date_max = context->shared_data.load();
                while (true) {
                    auto proposed_new_max = heap[0].first;
                    if (up_to_date_max <= proposed_new_max) {
                        break;
                    }
                    if (context->shared_data.compare_exchange_weak(up_to_date_max, proposed_new_max)) {
                        break;
                    }
                }
            }
        }
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void fill_dist_to_anchor_buffer_worker(ParallelContext<T> *context, std::size_t thread_id, std::span<hyperspheres::structures::IdTaggedDistance<T>> buffer, std::size_t start, std::size_t end) {
        auto func = context->dist_func;
        const auto cached_max_dists = context->vector_input1.subspan(start, end - start);
        std::size_t n_outputs = 0;
        for (const auto [_, i] : cached_max_dists) {
            const auto dist = func(context->vector_input2, std::span<T>{context->vector_input3.data() + i * context->db_storage.vector_dim, context->db_storage.vector_dim});
            buffer[n_outputs++] = {dist, i};
        }
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void find_knn_worker(ParallelContext<T> *context, std::size_t thread_id, std::span<hyperspheres::structures::IdTaggedDistance<T>> heap, std::size_t &n_outputs) {
        const auto dim = context->db_storage.vector_dim;
        const auto max_morsel = context->max_morsel;
        const auto step = context->morsel_step;
        const auto k = context->scalar_input1;
        const float beta = context->pruning_beta;
        auto func = context->dist_func;

        std::size_t n_p = 0;

        while (true) {
            const auto morsel = context->next_morsel.fetch_add(step);
            if (morsel >= max_morsel) {
                context->n_pruned.fetch_add(n_p);
                return;
            }
            const auto current_global_max = context->shared_data.load();
            for (std::size_t current = morsel; current < std::min(max_morsel, step + morsel); ++current) {
                const auto [dist_to_anchor, page_idx] = context->vector_input1[current];
                const auto max = hyperspheres::execution::storage::get_max_dist_at<T>(context->db_storage, page_idx);
                if (context->prune_check_func(dist_to_anchor, max, current_global_max, beta)) {
                    n_p++;
                    continue;
                }
                if (n_outputs < k) {
                    const std::size_t vector_id = hyperspheres::execution::storage::get_vector_id(context->db_storage, page_idx, 0);
                    heap[n_outputs++] = {dist_to_anchor, vector_id};
                    std::ranges::push_heap(heap.begin(), heap.begin() + n_outputs);
                } else {
                    if (context->prune_check_func(dist_to_anchor, max, heap[0].first, beta)) {
                        n_p++;
                        continue;
                    }
                    if (heap[0].first > dist_to_anchor) {
                        const std::size_t vector_id = hyperspheres::execution::storage::get_vector_id(context->db_storage, page_idx, 0);
                        std::ranges::pop_heap(heap.begin(), heap.begin() + n_outputs);
                        heap[k-1] = {dist_to_anchor, vector_id};
                        std::ranges::push_heap(heap.begin(), heap.begin() + n_outputs);
                    }
                }
                auto view = hyperspheres::execution::storage::get_array_view_of_page<T>(context->db_storage, page_idx);
                for (std::size_t i = 1; i < view.extent(0); i++) {
                    const auto dist = func(std::span<T>{view.data_handle() + i * view.extent(1), view.extent(1) }, context->vector_input2);
                    const auto vector_id = hyperspheres::execution::storage::get_vector_id(context->db_storage, page_idx, i);
                    if (n_outputs == k) {
                        if (heap[0].first > dist) {
                            std::ranges::pop_heap(heap.begin(), heap.begin() + n_outputs);
                            heap[k-1] = {dist, vector_id};
                            std::ranges::push_heap(heap.begin(), heap.begin() + n_outputs);
                        }
                    } else {
                        heap[n_outputs++] = {dist, vector_id};
                        std::ranges::push_heap(heap.begin(), heap.begin() + n_outputs);
                    }
                }
            }
            if (heap[0].first < current_global_max && n_outputs == k) {
                auto up_to_date_max = context->shared_data.load();
                while (true) {
                    auto proposed_new_max = heap[0].first;
                    if (up_to_date_max <= proposed_new_max) {
                        break;
                    }
                    if (context->shared_data.compare_exchange_weak(up_to_date_max, proposed_new_max)) {
                        break;
                    }
                }
            }
        }
    }

}

export namespace hyperspheres::execution::parallel {

    template <typename T>
    void hyperspheres_worker(ParallelContext<T>* context, std::size_t thread_id, const std::size_t buffer_size, std::atomic<std::size_t>* global_task_id, std::atomic<bool> *job_done) {
        std::size_t task_id {0};
        auto local_buffer = context->thread_local_output_buffer1.subspan(thread_id * buffer_size, buffer_size);
        while (true) {
            global_task_id->wait(task_id, std::memory_order_acquire);
            task_id++;
            auto &n_outputs = context->vector_output_buffer3[thread_id];
            n_outputs = 0;
            switch (context->task) {
                case ParallelTask::Exit:
                    job_done->store(true, std::memory_order_release);
                    job_done->notify_one();
                    return;
                case ParallelTask::FindKnn:
                    workers::find_knn_worker(context, thread_id, local_buffer, n_outputs);
                    break;
                case ParallelTask::BatchPrefill:
                    workers::batch_prefill_worker(context, thread_id, local_buffer, n_outputs);
                    break;
                case ParallelTask::FillDistToAnchor: {
                    if (thread_id < context->scalar_input1) {
                        const auto start = (context->morsel_step + 1) * thread_id;
                        const auto end = start + context->morsel_step + 1;
                        const auto dists = context->vector_output_buffer4.subspan(start, end - start);
                        workers::fill_dist_to_anchor_buffer_worker(context, thread_id, dists, start, end);
                    } else {
                        const auto start = (context->morsel_step + 1) * context->scalar_input1 + context->morsel_step * (thread_id - context->scalar_input1);
                        const auto unchecked_end = start + context->morsel_step;
                        const auto end = unchecked_end > context->max_morsel ? context->max_morsel : unchecked_end;
                        const auto dists = context->vector_output_buffer4.subspan(start, end - start);
                        workers::fill_dist_to_anchor_buffer_worker(context, thread_id, dists, start, end);
                    }
                }
                    break;
                default:
                    break;
            }
            job_done->store(true, std::memory_order_release);
            job_done->notify_one();
        }
    }

    template <typename T>
    void submit_and_block(ThreadPool<T> *pool) {
        for (std::size_t i = 0; i < pool->n_threads; i++) {
            pool->thread_job_done[i]->store(false, std::memory_order_release);
        }
        pool->global_task_id->fetch_add(1,std::memory_order_release);
        pool->global_task_id->notify_all();
        for (std::size_t i = 0; i < pool->n_threads; i++) {
            pool->thread_job_done[i]->wait(false, std::memory_order_acquire);
        }
    }

}
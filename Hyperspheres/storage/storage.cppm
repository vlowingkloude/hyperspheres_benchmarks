module;

#include <cstddef>
#include <string>
#include <vector>

export module storage;

import memorymap;

namespace hyperspheres::storage::page {

    struct Page {
        std::size_t pagesize;
        std::byte* page_addr;
        std::size_t *n_vectors;
        std::byte *max_dist;

        Page() = delete;

        explicit Page(std::size_t pagesize, std::byte* page_addr);

        ~Page();

        Page(const Page&) = delete;
        Page& operator=(const Page&) = delete;

        Page(Page&& other) noexcept;
    };

}

export namespace hyperspheres::storage {

    struct MappedDataFile {
        std::string filename;
        hyperspheres::memorymap::MappedFile mapped_file;
        std::size_t pagesize;
        std::vector<page::Page> pages;

        MappedDataFile() = delete;

        explicit MappedDataFile(const std::string& path, std::size_t p_size, std::size_t f_size = 0);

        MappedDataFile(const MappedDataFile&) = delete;
        MappedDataFile& operator=(const MappedDataFile&) = delete;

        MappedDataFile(MappedDataFile&& other) noexcept;
    };

    struct Storage {
        std::string path;
        std::vector<MappedDataFile> files;
        std::size_t n_files;
        std::size_t page_size;
        std::size_t file_size;
        std::size_t id_offset;
        std::size_t vector_offset;
        std::size_t n_pages;
        std::size_t element_size;
        std::size_t vector_dim;
        std::size_t max_num_vectors_per_page;
    };

}
module;

#include <cstddef>
#include <utility>
#include <numeric>

module storage;

namespace hyperspheres::storage::page {

    Page::Page(const std::size_t pagesize, std::byte *page_addr) : pagesize(pagesize), page_addr(page_addr) {
        // by definition, the first size_t bytes are the number of vectors
        n_vectors = std::bit_cast<std::size_t *, std::byte *>(page_addr);
        max_dist = page_addr + sizeof(std::size_t);
    }

    Page::~Page() {
        page_addr = nullptr;
    }

    Page::Page(Page &&other) noexcept : pagesize(other.pagesize), page_addr(other.page_addr), n_vectors(other.n_vectors), max_dist(other.max_dist) {
        other.page_addr = nullptr;
    }

}

module;

#include <cstddef>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>

module storage;

namespace hyperspheres::storage {

    MappedDataFile::MappedDataFile(const std::string &path, const std::size_t p_size, const std::size_t f_size)
        : filename(path), mapped_file(path, hyperspheres::memorymap::MmapMode::ReadWrite, f_size), pagesize(p_size)
    {
        const std::size_t n_pages = mapped_file.filesize / pagesize; // todo: better check the divide here
        pages.reserve(n_pages);
        for (auto i = 0; i < n_pages; i++) {
            pages.emplace_back(pagesize, mapped_file.addr + i * pagesize);
        }
    }

    MappedDataFile::MappedDataFile(MappedDataFile &&other) noexcept
        : filename(std::move(other.filename)), mapped_file(std::move(other.mapped_file)), pagesize(other.pagesize), pages(std::move(other.pages))
    {
        other.pagesize = 0;
    }
}

module;

#include <cstddef>
#include <utility>
#include <memory>

export module memorymap;

export namespace hyperspheres::memorymap {

    enum MmapMode {
        Read, Write, ReadWrite
    };

    struct MappedFile {
        int fd;
        std::byte *addr;
        std::size_t filesize;

        MappedFile() = delete;

        explicit MappedFile(const std::string& filename, MmapMode mode = MmapMode::Read, std::size_t create_file_size = 0);
        ~MappedFile();
        MappedFile(const MappedFile&) = delete;

        MappedFile(MappedFile&& other) noexcept;
    };

}
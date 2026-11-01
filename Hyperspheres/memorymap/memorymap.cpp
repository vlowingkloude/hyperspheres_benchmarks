module;

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <memory>
#include <fstream>

#ifdef _WIN32
// #include <windows.h>
// #include <fcntl.h>
// #include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

module memorymap;

namespace hyperspheres::memorymap {

    MappedFile::MappedFile(const std::string& filename, const MmapMode mode, const std::size_t create_file_size) {
#ifdef _WIN32
        throw std::runtime_error("MappedDatabaseFile not implemented for Windows.");
        // HANDLE handle = CreateFileA(filename.data(), GENERIC_READ | GENERIC_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        // if (handle == INVALID_HANDLE_VALUE) {
        //     throw std::runtime_error("failed to open file");
        // }
        // fd = handle;
        // LARGE_INTEGER file_size;
        // GetFileSizeEx(fd, &file_size);
        // size = static_cast<std::size_t>(file_size.QuadPart);
        // HANDLE map = CreatFileMappingA(fd, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        // addr = MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        // CloseHandle(map);
#else
        // const auto posix_mode = mode == MmapMode::Read ? O_RDONLY : mode == MmapMode::Write ? O_WRONLY : O_RDWR;
        const auto posix_mode = mode == MmapMode::Read ? O_RDONLY : O_RDWR;
        const auto posix_prot = mode == MmapMode::Read ? PROT_READ : mode == MmapMode::Write ? PROT_WRITE : PROT_READ | PROT_WRITE;
        if (mode != MmapMode::Read && create_file_size > 0) {
            fd = open(filename.data(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            posix_fallocate(fd, 0, static_cast<long>(create_file_size));
            close(fd);
        }
        fd = open(filename.data(), posix_mode);
        if (fd < 0) {
            throw std::runtime_error("failed to open file");
        }
        struct stat stat_buf {};
        fstat(fd, &stat_buf);
        filesize = static_cast<std::size_t>(stat_buf.st_size);
        // should be fine to just use map_shared here
        addr = static_cast<std::byte *>(mmap(nullptr, filesize, posix_prot, MAP_SHARED, fd, 0));
        posix_madvise(addr, filesize, POSIX_MADV_WILLNEED);
#endif
    }

    MappedFile::~MappedFile() {
        if (addr != nullptr) {
#ifdef _WIN32
            UnmapViewOfFile(addr);
            CloseHandle(fd);
#else
            msync(addr, filesize, MS_SYNC);
            munmap(addr, filesize);
            close(fd);
#endif
            addr = nullptr;
            fd = -1;
            filesize = 0;
        }
    }

    MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd(other.fd), addr(other.addr), filesize(other.filesize) {
        other.addr = nullptr;
        other.filesize = 0;
        other.fd = -1;
    }

}
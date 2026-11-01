#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            const int error = errno;
            ::close(fd_);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        size_ = static_cast<std::size_t>(st.st_size);
        data_ = static_cast<const std::byte*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            const int error = errno;
            ::close(fd_);
            data_ = nullptr;
            throw std::system_error(error, std::generic_category(), "mmap " + path.string());
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    ~MappedFile() {
        if (data_ != nullptr) {
            ::munmap(const_cast<std::byte*>(data_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    [[nodiscard]] const std::byte* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }

private:
    int fd_ = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

struct FvecsView {
    explicit FvecsView(const std::filesystem::path& path) : file(path) {
        if (file.size() < sizeof(std::int32_t)) {
            throw std::runtime_error(path.string() + ": empty or truncated fvecs file");
        }
        std::memcpy(&dim, file.data(), sizeof(dim));
        if (dim <= 0) {
            throw std::runtime_error(path.string() + ": invalid dimension");
        }
        row_bytes = sizeof(std::int32_t) + static_cast<std::size_t>(dim) * sizeof(float);
        if (file.size() % row_bytes != 0) {
            throw std::runtime_error(path.string() + ": inconsistent fvecs file size");
        }
        rows = file.size() / row_bytes;
    }

    [[nodiscard]] const float* row(const std::size_t index) const {
        return reinterpret_cast<const float*>(
            file.data() + index * row_bytes + sizeof(std::int32_t));
    }

    MappedFile file;
    std::int32_t dim = 0;
    std::size_t rows = 0;
    std::size_t row_bytes = 0;
};

void copy(const FvecsView& source, float* destination) {
        const std::size_t dim = static_cast<std::size_t>(source.dim);

    for (std::size_t row = 0; row < source.rows; ++row) {
        const float* input = source.row(row);
        float* output = destination + row * dim;
        for (std::size_t col = 0; col < dim; ++col) {
                output[col] = input[col];
        }
    }
}

void copy_normalized(const FvecsView& source, float* destination) {
    const std::size_t dim = static_cast<std::size_t>(source.dim);

#pragma omp parallel for schedule(static)
    for (std::size_t row = 0; row < source.rows; ++row) {
        const float* input = source.row(row);
        float* output = destination + row * dim;
        float squared_norm = 0.0F;
#pragma omp simd reduction(+ : squared_norm)
        for (std::size_t col = 0; col < dim; ++col) {
            squared_norm += input[col] * input[col];
        }
        if (!(squared_norm > 0.0F) || !std::isfinite(squared_norm)) {
#pragma omp simd
            for (std::size_t col = 0; col < dim; ++col) {
                output[col] = input[col];
            }
        }
        const float inverse_norm = 1.0F / std::sqrt(squared_norm);
#pragma omp simd
        for (std::size_t col = 0; col < dim; ++col) {
            output[col] = input[col] * inverse_norm;
        }
    }
}

struct Neighbor {
    float squared_distance;
    std::int32_t id;

    bool operator<(const Neighbor& other) const {
        if (squared_distance != other.squared_distance) {
            return squared_distance < other.squared_distance;
        }
        return id < other.id;
    }
};

float normalized_l2_squared(const float* lhs, const float* rhs, const std::size_t dim) {
    float sums[8] {};
    std::size_t col = 0;
    for (; col + 7 < dim; col += 8) {
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const float difference = lhs[col + lane] - rhs[col + lane];
            sums[lane] += difference * difference;
        }
    }
    float distance = ((sums[0] + sums[4]) + (sums[1] + sums[5]))
                   + ((sums[2] + sums[6]) + (sums[3] + sums[7]));
    for (; col < dim; ++col) {
        const float difference = lhs[col] - rhs[col];
        distance += difference * difference;
    }
    return distance;
}

float l2_squared(const float* lhs, const float* rhs, const std::size_t dim) {
    float sums[8] {};
    std::size_t col = 0;
    for (; col + 7 < dim; col += 8) {
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const float difference = lhs[col + lane] - rhs[col + lane];
            sums[lane] += difference * difference;
        }
    }
    float distance = ((sums[0] + sums[4]) + (sums[1] + sums[5]))
                   + ((sums[2] + sums[6]) + (sums[3] + sums[7]));
    for (; col < dim; ++col) {
        const float difference = lhs[col] - rhs[col];
        distance += difference * difference;
    }
    return distance;
}

std::vector<Neighbor> find_top_k(
    const float* base,
    const std::size_t base_rows,
    const float* query,
    const std::size_t dim,
    const std::size_t k) {
    std::vector<Neighbor> heap;
    heap.reserve(k);
    for (std::size_t row = 0; row < base_rows; ++row) {
        const Neighbor candidate {
            l2_squared(base + row * dim, query, dim),
            static_cast<std::int32_t>(row),
        };
        if (heap.size() < k) {
            heap.push_back(candidate);
            std::push_heap(heap.begin(), heap.end());
        } else if (candidate < heap.front()) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = candidate;
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

void write_ivecs(
    const std::filesystem::path& output_path,
    const std::vector<std::vector<Neighbor>>& neighbors,
    const std::size_t k) {
    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot open output " + output_path.string());
    }
    const auto dimension = static_cast<std::int32_t>(k);
    for (const auto& row : neighbors) {
        output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        for (const Neighbor& neighbor : row) {
            output.write(reinterpret_cast<const char*>(&neighbor.id), sizeof(neighbor.id));
        }
    }
    if (!output) {
        throw std::runtime_error("failed while writing " + output_path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "usage: " << argv[0]
                      << " <base.fvecs> <query.fvecs> <groundtruth.ivecs> [k=100]\n";
            return 2;
        }

        const std::filesystem::path base_path = argv[1];
        const std::filesystem::path query_path = argv[2];
        const std::filesystem::path output_path = argv[3];
        const std::size_t k = argc == 5 ? std::stoull(argv[4]) : 100;

        const FvecsView base_view(base_path);
        const FvecsView query_view(query_path);
        if (base_view.dim != query_view.dim) {
            throw std::runtime_error("base and query dimensions differ");
        }
        if (k == 0 || k > base_view.rows) {
            throw std::runtime_error("k must be in [1, base_count]");
        }
        if (base_view.rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("ivecs cannot represent base vector IDs");
        }

        const std::size_t dim = static_cast<std::size_t>(base_view.dim);
        std::cerr << "base=" << base_view.rows << " query=" << query_view.rows
                  << " dim=" << dim << " k=" << k;
#ifdef _OPENMP
        std::cerr << " threads=" << omp_get_max_threads();
#endif
        std::cerr << '\n';

        auto normalized_base = std::make_unique_for_overwrite<float[]>(base_view.rows * dim);
        auto normalized_queries = std::make_unique_for_overwrite<float[]>(query_view.rows * dim);
        //copy_normalized(base_view, normalized_base.get());
        //copy_normalized(query_view, normalized_queries.get());
        copy(base_view, normalized_base.get());
        copy(query_view, normalized_queries.get());
        std::cerr << "normalization complete\n";

        std::vector<std::vector<Neighbor>> result(query_view.rows);
        std::atomic<std::size_t> completed = 0;
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t query = 0; query < query_view.rows; ++query) {
            result[query] = find_top_k(
                normalized_base.get(), base_view.rows,
                normalized_queries.get() + query * dim, dim, k);
            const std::size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % 10 == 0 || done == query_view.rows) {
#pragma omp critical
                std::cerr << "completed " << done << '/' << query_view.rows << '\n';
            }
        }

        write_ivecs(output_path, result, k);
        std::cerr << "wrote " << output_path << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
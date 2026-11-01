#ifndef SPHERES_BENCHMARK_FVECS_STREAM_H
#define SPHERES_BENCHMARK_FVECS_STREAM_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

class FvecsBatchReader {
public:
    FvecsBatchReader(std::string path, const std::size_t dim)
        : path_(std::move(path)), dim_(dim), input_(path_, std::ios::binary) {
        if (!input_) {
            throw std::runtime_error("failed to open " + path_);
        }
    }

    std::span<const float> read(const std::size_t requested_rows) {
        buffer_.resize(requested_rows * dim_);
        for (std::size_t row = 0; row < requested_rows; ++row) {
            std::int32_t file_dim = 0;
            input_.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
            if (!input_ || file_dim != static_cast<std::int32_t>(dim_)) {
                throw std::runtime_error("invalid dimension or truncated file: " + path_);
            }
            input_.read(
                reinterpret_cast<char*>(buffer_.data() + row * dim_),
                static_cast<std::streamsize>(dim_ * sizeof(float)));
            if (!input_) {
                throw std::runtime_error("truncated vector data: " + path_);
            }
        }
        return std::span<const float>(buffer_.data(), buffer_.size());
    }

private:
    std::string path_;
    std::size_t dim_;
    std::ifstream input_;
    std::vector<float> buffer_;
};

#endif

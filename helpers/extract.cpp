#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <span>
#include <algorithm>
#include <bit>
#include <limits>
#include <cstdlib>
#include <charconv>
#include <system_error>
#include <random>

#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

void write_fvecs(const std::string& path, std::vector<std::vector<float>> &vecs, std::size_t rows, std::size_t dim) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot open output " + path);
    }
    const auto dimension = static_cast<std::int32_t>(dim);
    for (const auto& row : vecs) {
        output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        output.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * dimension);
    }
    if (!output) {
        throw std::runtime_error("failed while writing " + path);
    }
}

std::vector<std::vector<float>> read_file_as_vectors(const std::string& path, std::size_t rows, std::size_t dim) {
    std::vector<std::vector<float>> data(rows);
    std::ifstream ins(path, std::ios::binary);
    if (!ins) {
        throw std::runtime_error("failed to open " + path);
    }
    int32_t d;
    for (auto i = 0; i < rows; ++i) {
        std::vector<float> temp (dim);
        ins.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!ins || d != static_cast<std::int32_t>(dim)) {
            throw std::runtime_error("invalid dimension or truncated file: " + path);
        }
        ins.read(std::bit_cast<char *, float *>(temp.data()), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!ins) {
            throw std::runtime_error("truncated vector data: " + path);
        }
        data[i] = std::move(temp);
    }
    return data;
}

int main(int argc, char *argv[]) {
    auto old_vecs = read_file_as_vectors(argv[1], std::stol(argv[2]), std::stol(argv[3]));
    write_fvecs(argv[4], old_vecs, std::stol(argv[2]), std::stol(argv[5]));
    return 0;
}
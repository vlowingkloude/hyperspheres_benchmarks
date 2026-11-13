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

    float dot_product_naive(const std::vector<float> &x, const std::vector<float> &y) {
        float dot_product {0};
        std::size_t i;
        float xy0, xy1, xy2, xy3, xy4, xy5, xy6, xy7;
        xy0 = xy1 = xy2 = xy3 = xy4 = xy5 = xy6 = xy7 = 0;
        for (i = 0; i + 7 < x.size(); i += 8) {
            xy0 = x[i + 0] * y[i + 0] + xy0;
            xy1 = x[i + 1] * y[i + 1] + xy1;
            xy2 = x[i + 2] * y[i + 2] + xy2;
            xy3 = x[i + 3] * y[i + 3] + xy3;
            xy4 = x[i + 4] * y[i + 4] + xy4;
            xy5 = x[i + 5] * y[i + 5] + xy5;
            xy6 = x[i + 6] * y[i + 6] + xy6;
            xy7 = x[i + 7] * y[i + 7] + xy7;
        }
        dot_product = ((xy0 + xy4) + (xy1 + xy5)) + ((xy2 + xy6) + (xy3 + xy7));
        if (i != x.size()) [[unlikely]] {
            for (;i < x.size(); ++i) {
                dot_product = x[i] * y[i] + dot_product;
            }
        }
        return dot_product;
    }

    void normalize(std::vector<float> &x) {
        float norm_x = dot_product_naive(x, x);
        if (norm_x == 0) [[unlikely]] {
            // todo: maybe better throw
            return;
        }
        norm_x = std::sqrt(norm_x);
        std::size_t i;
        for (i = 0; i + 7 < x.size(); i += 8) {
            x[i + 0] = x[i + 0] / norm_x;
            x[i + 1] = x[i + 1] / norm_x;
            x[i + 2] = x[i + 2] / norm_x;
            x[i + 3] = x[i + 3] / norm_x;
            x[i + 4] = x[i + 4] / norm_x;
            x[i + 5] = x[i + 5] / norm_x;
            x[i + 6] = x[i + 6] / norm_x;
            x[i + 7] = x[i + 7] / norm_x;
        }
        if (i != x.size()) [[unlikely]] {
            for (;i < x.size(); ++i) {
                x[i] = x[i] / norm_x;
            }
        }
    }

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
    for(auto &v : old_vecs) {
        normalize(v);
    }
    write_fvecs(argv[4], old_vecs, std::stol(argv[2]), std::stol(argv[3]));
    return 0;
}
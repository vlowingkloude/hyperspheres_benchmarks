#include <exception>
#include <iostream>

#include "engines.h"

int main(int argc, char **argv) {

    try {
        BenchmarkConfig config = parse_argv(argc, argv);
        if (config.engine_name == "faiss") {
            faiss_benchmark::run(config);
        } else {
            throw std::invalid_argument("unknown engine: " + config.engine_name);
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}

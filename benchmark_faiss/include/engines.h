#ifndef SPHERES_BENCHMARK_ENGINES_H
#define SPHERES_BENCHMARK_ENGINES_H

#include "benchmark_config.h"

namespace baseline { void run(const BenchmarkConfig& config); }
namespace faiss_benchmark { void run(const BenchmarkConfig& config); }
namespace hnswlib_benchmark { void run(const BenchmarkConfig& config); }
namespace spheres_benchmark { void run(const BenchmarkConfig& config); }
namespace duckdb_benchmark { void run(const BenchmarkConfig& config); }
namespace pgvector_benchmark { void run(const BenchmarkConfig& config); }

#endif

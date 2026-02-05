// src/runner.cu
//
// Benchmark SGEMM kernels vs cuBLAS (strict FP32) with optional CSV logging.
//
// Usage:
//   ./runner                          # runs: cuBLAS + all kernels, M=N=K=1024
//   ./runner naive 4096 4096 4096     # runs: cuBLAS + naive, prints ms/GFLOP/s/% + error
//   ./runner all 4096 4096 4096       # runs: cuBLAS + all kernels
//
// Optional logging:
//   ./runner naive 4096 4096 4096 --log results.csv
//   ./runner all  4096 4096 4096 --log results.csv
//
// Kernels expected in kernels.cuh:
//   sgemm_naive
//   sgemm_global_mem_coalesce<BS>
//   sgemm_shared_mem_block<BS>
//   sgemm_1D_blocktiling<BM,BN,BK,TM>
//   sgemm_2D_blocktiling<BM,BN,BK,TM,TN>
//   sgemm_vectorize<BM,BN,BK,TM,TN>
//
// Notes:
// - Assumes row-major A(MxK), B(KxN), C(MxN).
// - cuBLAS is column-major; we compute C^T = B^T * A^T so memory matches row-major C.
// - "Strict FP32" baseline: CUBLAS_DEFAULT_MATH (TF32 tensor-op mode NOT enabled).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels.cuh"

// ------------------ helpers ------------------
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call);                                       \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(err));         \
    std::exit(1);                                                 \
  }                                                               \
} while(0)

#define CUBLAS_CHECK(call) do {                                   \
  cublasStatus_t st = (call);                                     \
  if (st != CUBLAS_STATUS_SUCCESS) {                              \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n",                   \
            __FILE__, __LINE__, (int)st);                         \
    std::exit(1);                                                 \
  }                                                               \
} while(0)

static inline double gflops_from_ms(int M, int N, int K, float ms) {
  const double flops = 2.0 * (double)M * (double)N * (double)K;
  return (flops / (ms / 1000.0)) / 1e9;
}

static inline void usage(const char* prog) {
  std::printf(
    "Usage:\n"
    "  %s [kernel] [M N K] [--log file.csv]\n\n"
    "kernel: naive | coalesced | shared | 1d | 2d | vec | all\n"
    "defaults: kernel=all, M=N=K=1024\n\n"
    "Examples:\n"
    "  %s\n"
    "  %s naive 4096 4096 4096\n"
    "  %s all   4096 4096 4096 --log results.csv\n",
    prog, prog, prog, prog
  );
}

static bool file_exists(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

static void append_csv_header_if_needed(const char* path) {
  if (!path) return;
  if (file_exists(path)) return;
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::fprintf(stderr, "Failed to open log file for writing: %s\n", path);
    std::exit(1);
  }
  std::fprintf(f,
    "gpu,kernel,M,N,K,alpha,beta,math_mode,ms,gflops,pct_of_cublas,max_abs,max_rel\n");
  std::fclose(f);
}

static void append_csv_row(const char* path,
                           const char* gpu_name,
                           const char* kernel,
                           int M,int N,int K,
                           float alpha,float beta,
                           const char* math_mode,
                           float ms,double gflops,double pct,
                           float max_abs,float max_rel) {
  if (!path) return;
  FILE* f = std::fopen(path, "ab");
  if (!f) {
    std::fprintf(stderr, "Failed to open log file for append: %s\n", path);
    std::exit(1);
  }
  std::fprintf(f, "\"%s\",%s,%d,%d,%d,%.8g,%.8g,%s,%.6f,%.6f,%.6f,%.6e,%.6e\n",
               gpu_name, kernel, M,N,K, alpha,beta, math_mode,
               ms, gflops, pct, max_abs, max_rel);
  std::fclose(f);
}

template <typename LaunchFn>
static float time_kernel(const char* name,
                         LaunchFn&& launch,
                         float* dC, size_t bytesC,
                         cudaEvent_t start, cudaEvent_t stop,
                         int warmup, int iters) {
  CUDA_CHECK(cudaMemset(dC, 0, bytesC));

  for (int i = 0; i < warmup; i++) {
    launch();
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) {
    launch();
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  return total_ms / (float)iters;
}

static void compute_error_stats(const std::vector<float>& ref,
                                const std::vector<float>& got,
                                float& max_abs,
                                float& max_rel) {
  max_abs = 0.0f;
  max_rel = 0.0f;
  const size_t n = ref.size();
  for (size_t i = 0; i < n; i++) {
    const float r = ref[i];
    const float g = got[i];
    const float abs_err = std::abs(g - r);
    max_abs = std::max(max_abs, abs_err);
    const float denom = std::max(1e-8f, std::abs(r));
    const float rel_err = abs_err / denom;
    max_rel = std::max(max_rel, rel_err);
  }
}

// Strict FP32 cuBLAS: compute C^T (N x M) = B^T (N x K) * A^T (K x M)
// in column-major, which matches row-major C in memory layout.
static float time_cublas_strict_fp32(cublasHandle_t handle,
                                     int M, int N, int K,
                                     float alpha, const float* dA, const float* dB,
                                     float beta, float* dC, size_t bytesC,
                                     cudaEvent_t start, cudaEvent_t stop,
                                     int warmup, int iters) {
  return time_kernel("cuBLAS", [&]{
    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            dB, N,
                            dA, K,
                            &beta,
                            dC, N));
  }, dC, bytesC, start, stop, warmup, iters);
}

// ------------------ "registry" (minimal) ------------------
struct KernelSpec {
  const char* name;
  // Launch once (for correctness) and launch for timing
  // (we just reuse the same lambda for both)
  std::function<void()> launch;
};
// ----------------------------------------------------------

int main(int argc, char** argv) {
  std::string which = "all";
  int M = 1024, N = 1024, K = 1024;
  const char* log_path = nullptr;

  // Parse args:
  //   runner [kernel] [M N K] [--log file.csv]
  // Keep it simple & permissive.
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else if (std::strcmp(argv[i], "--log") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--log requires a path\n");
        return 1;
      }
      log_path = argv[i + 1];
      i++;
    } else if (which == "all" && i == 1) {
      // first positional arg: kernel name (optional)
      which = argv[i];
    } else if (i + 2 < argc) {
      // try parse M N K at this position
      M = std::atoi(argv[i]);
      N = std::atoi(argv[i + 1]);
      K = std::atoi(argv[i + 2]);
      i += 2;
    } else {
      // unknown trailing
    }
  }

  if (M <= 0 || N <= 0 || K <= 0) {
    std::fprintf(stderr, "Invalid sizes M N K\n");
    return 1;
  }

  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 5;
  const int iters  = 50;

  // GPU info for logging
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  if (log_path) append_csv_header_if_needed(log_path);

  std::printf("Selected: %s   M=%d N=%d K=%d\n", which.c_str(), M, N, K);
  std::printf("GPU: %s\n", prop.name);

  // Host buffers
  std::vector<float> A((size_t)M * K, 1.0f);
  std::vector<float> B((size_t)K * N, 1.0f);

  // Device buffers
  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // cuBLAS handle (strict FP32)
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  const char* math_mode = "CUBLAS_DEFAULT_MATH";

  const size_t bytesC = (size_t)M * N * sizeof(float);

  // Build cuBLAS reference output (single run)
  std::vector<float> C_ref((size_t)M * N, 0.0f);
  {
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            dB, N,
                            dA, K,
                            &beta,
                            dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C_ref.data(), dC, bytesC, cudaMemcpyDeviceToHost));
  }

  // Time cuBLAS baseline
  float cublas_ms = time_cublas_strict_fp32(handle, M,N,K, alpha, dA,dB, beta, dC, bytesC,
                                           start, stop, warmup, iters);
  double cublas_gflops = gflops_from_ms(M,N,K, cublas_ms);

  std::printf("%-10s time = %.4f ms   perf = %.2f GFLOP/s\n",
              "cuBLAS", cublas_ms, cublas_gflops);

  append_csv_row(log_path, prop.name, "cuBLAS", M,N,K, alpha,beta, math_mode,
                 cublas_ms, cublas_gflops, 100.0, 0.0f, 0.0f);

  // ---- define kernel launch configs here (small, centralized) ----
  std::vector<KernelSpec> kernels;
  kernels.reserve(8);

  // naive (2D block)
  {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, (int)block.x), CEIL_DIV(N, (int)block.y));
    kernels.push_back({"naive", [=]{
      sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }});
  }

  // coalesced (1D block)
  {
    constexpr int BS = 32;
    dim3 block(BS * BS, 1, 1);
    dim3 grid(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
    kernels.push_back({"coalesced", [=]{
      sgemm_global_mem_coalesce<BS><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }});
  }

  // shared (1D block)
  {
    constexpr int BS = 32;
    dim3 block(BS * BS, 1, 1);
    dim3 grid(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
    kernels.push_back({"shared", [=]{
      sgemm_shared_mem_block<BS><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }});
  }

  // 1d blocktiling (example params)
  {
    // Ensure your kernel uses Bs correctly and its asserts match your design.
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block(BM * BK, 1, 1);                   // 512 threads
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));  // x=col tiles, y=row tiles
    kernels.push_back({"1d", [=]{
      sgemm_1D_blocktiling<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }});
  }

  // 2d blocktiling (example params)
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int threads = (BM * BN) / (TM * TN); // 256
    dim3 block(threads, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    kernels.push_back({"2d", [=]{
      sgemm_2D_blocktiling<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }});
  }

  // vectorized (example params)
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int threads = (BM * BN) / (TM * TN); // 256
    dim3 block(threads, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    kernels.push_back({"vec", [=]{
      sgemm_vectorize<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }});
  }

  auto find_kernel = [&](const std::string& name) -> KernelSpec* {
    for (auto& kspec : kernels) {
      if (name == kspec.name) return &kspec;
    }
    return nullptr;
  };

  auto run_and_report = [&](KernelSpec& ks) {
    // correctness run
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    ks.launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_got((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(C_got.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    float max_abs=0.0f, max_rel=0.0f;
    compute_error_stats(C_ref, C_got, max_abs, max_rel);
    std::printf("  check %-10s max_abs=%.3e  max_rel=%.3e\n",
                ks.name, max_abs, max_rel);

    // timed run
    float ms = time_kernel(ks.name, ks.launch, dC, bytesC, start, stop, warmup, iters);
    double gf = gflops_from_ms(M,N,K, ms);
    double pct = (cublas_gflops > 0.0) ? (100.0 * gf / cublas_gflops) : 0.0;

    std::printf("%-10s time = %.4f ms   perf = %.2f GFLOP/s   (%.2f%% of cuBLAS)\n",
                ks.name, ms, gf, pct);

    append_csv_row(log_path, prop.name, ks.name, M,N,K, alpha,beta, math_mode,
                   ms, gf, pct, max_abs, max_rel);
  };

  // selection
  if (which == "all") {
    for (auto& ks : kernels) run_and_report(ks);
  } else {
    KernelSpec* ks = find_kernel(which);
    if (!ks) {
      std::fprintf(stderr, "Unknown kernel '%s'\n", which.c_str());
      usage(argv[0]);
      return 1;
    }
    run_and_report(*ks);
  }

  // cleanup
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}

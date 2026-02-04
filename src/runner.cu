#include <cstdio>
#include <cstdlib>
#include <vector>

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

// --------------------------------------------

int main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 1024;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  float alpha = 1.0f, beta = 0.0f;

  std::vector<float> A((size_t)M * K, 1.0f);
  std::vector<float> B((size_t)K * N, 1.0f);
  std::vector<float> C((size_t)M * N, 0.0f);

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice));

  // ----------------------------
  // Kernel 1: naive
  // ----------------------------
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, block.x), CEIL_DIV(N, block.y));

  CUDA_CHECK(cudaMemset(dC, 0, C.size() * sizeof(float)));

  for (int i = 0; i < 5; i++) {
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int iters = 50;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) {
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  ms /= iters;

  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = (flops / (ms / 1000.0)) / 1e9;

  printf("M=%d N=%d K=%d\n", M, N, K);
  printf("naive   time = %.4f ms\n", ms);
  printf("naive   perf = %.2f GFLOP/s\n", gflops);

  // ----------------------------
  // cuBLAS baseline
  // ----------------------------
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  CUDA_CHECK(cudaMemset(dC, 0, C.size() * sizeof(float)));

  for (int i = 0; i < 5; i++) {
    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            dB, N,
                            dA, K,
                            &beta,
                            dC, N));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) {
    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            dB, N,
                            dA, K,
                            &beta,
                            dC, N));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float cublas_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
  cublas_ms /= iters;

  double cublas_gflops = (flops / (cublas_ms / 1000.0)) / 1e9;

  printf("cuBLAS  time = %.4f ms\n", cublas_ms);
  printf("cuBLAS  perf = %.2f GFLOP/s\n", cublas_gflops);

  CUBLAS_CHECK(cublasDestroy(handle));

  // verification (cuBLAS result currently in dC)
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Verification: C[0]=%.2f (Expected %.2f)\n", C[0], (float)K * alpha);

  // cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}

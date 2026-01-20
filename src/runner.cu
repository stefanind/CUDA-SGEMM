#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "kernels.cuh"  // should include your sgemm_naive kernel

int ceil_div(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char** argv) {
  // 1) Read matrix sizes (defaults if none given)
  int M = 1024, N = 1024, K = 1024;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  float alpha = 1.0f, beta = 0.0f;

  // 2) Create input/output arrays on CPU (host)
  std::vector<float> A((size_t)M * K, 1.0f);
  std::vector<float> B((size_t)K * N, 1.0f);
  std::vector<float> C((size_t)M * N, 0.0f);

  // 3) Allocate arrays on GPU (device)
  float *dA, *dB, *dC;
  cudaMalloc(&dA, A.size() * sizeof(float));
  cudaMalloc(&dB, B.size() * sizeof(float));
  cudaMalloc(&dC, C.size() * sizeof(float));

  // 4) Copy CPU -> GPU
  cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);

  // 5) Choose how many threads + blocks to launch
  // Your naive kernel uses: x as row (0..M-1), y as col (0..N-1)
  dim3 block(32, 32);
  dim3 grid(ceil_div(M, block.x), ceil_div(N, block.y));

  // 6) Warmup (first launches can be slower)
  for (int i = 0; i < 5; i++)
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  cudaDeviceSynchronize();  // wait until GPU is done

  // 7) Time the kernel with CUDA events (GPU-side timing)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int iters = 50;
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);  // wait for the timed work to finish

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= iters;  // average time per kernel call

  // 8) Compute and print performance
  // GEMM does about 2*M*N*K floating point operations
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = (flops / (ms / 1000.0)) / 1e9;

  printf("M=%d N=%d K=%d\n", M, N, K);
  printf("time = %.4f ms\n", ms);
  printf("perf = %.2f GFLOP/s\n", gflops);

  // 9) Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}

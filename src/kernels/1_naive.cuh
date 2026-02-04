#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
simplest of matmul kernels

matrix sizes:
A = M x K -> A[row * K + k]
B = K x N -> B[k * N + col]
C = M x N -> C[row * N + col]

dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
dim3 blockDim(32, 32, 1);

sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C)

*/


__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, 
                            const float *B, float beta, float *C) {

    // the position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float temp = 0.0;
        for (int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        // C = alpha * (A @ B) + beta * C
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
      }
  }


/*

issue:
- does not access memory of B efficiently and writes to C inefficiently

solution for this issue in next kernel:
- reassign memory access of threads to efficiently write to C and gather memory contiguously from B

*/
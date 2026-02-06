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
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float temp = 0.0;
        for (int i = 0; i < K; ++i) {
            temp += A[row * K + i] * B[i * N + col];
        }
        // C = alpha * (A @ B) + beta * C
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
      }
  }


/*

(1) the next thread iterates across threadIdx.x
(2) threadIdx.x is mapped to rows
(3) therefore, each thread begins at its own row

(4) threadIdx.y is mapped to columns
(5) when threadIdx.x == blockDim.x - 1, threadIdx.y increments

(6) accessing formula of matrix: C[row * N + column], where N = col length of C
(7) threadidx.x = 0; threadidx.y = 0
(8) this thread calculates C[0 * N + 0] 
(9) this thread is working on C[0]


issues:
- threadIdx.x is mapped to the row index which causes varying threads to access
rows of A and C (strided access). This is very inefficient. 
- does not access memory of A and B efficiently and writes to C inefficiently

solution for this issue in next kernel:
- reassign memory access of threads to efficiently write to C and gather memory contiguously from B

*/
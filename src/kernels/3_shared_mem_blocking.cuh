#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>

__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B, float beta,
                                       float *C) {
    
    // row and col of respected block
    // note: not optimized for sequential accessing of B
    // (TODO in next kernel ^^)
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // store buffer in shared memory
    // shared memory is shared to all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // assign threads to memory within the inner row and col 
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting position in memory
    A += cRow * K * BLOCKSIZE;
    B += cCol * N;
    C += cRow * N * BLOCKSIZE + cCol * BLOCKSIZE;

    float temp = 0.0;
    for (bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // wait for all threads to cache data
        // needed for coalesced memory
        __syncthreads();

        // advance A and B to next tile within their row/col
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            temp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // sync to ensure all threads catch up
        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * temp + beta * C[threadRow * N + threadCol];

    
}


/*

issue:
- calls from shared memory twice for every fused multiply-addition,
therefore, it is memory-bound when a lot more compute is available.

solution for this issue in next kernel:
- each thread can perform more operations to use the available compute,
e.g., registers have a lot more memory per SM than SMEM.

*/
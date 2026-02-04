#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


// mark as constants for compile time 
template <const int BM, const int BN, const int BK, const int TM>

__global__ void sgemm_1D_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta, float *C) {
    // make the memory access sequential across each col on the row
    // i.e., memory increments via .x, not .y, so advancing via
    // blockIdx.x increments blocks based on how memory is assigned
    // before cCol advanced by blockIdx.y, now it advances by blockIdx.x
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float As[BK * BN];

    const int threadRow = threadIdx.x / BN;
    const int threadCol = threadIdx.x % BN;

    A += cRow * K * BM;
    B += cCol * BN;
    C += cRow * N * BM + cCol * BN;

    // check to ensure the tile count match thread count
    assert (BM * BK == blockDim.x);
    assert (BK * BN == blockDim.x);
    
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate SMEM buffer
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        // ensure all synced before continuing
        __syncthreads();

        // advance tiles
        A += BK;
        B += N * BK;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // store the reusable B value for each calc over the loop of A
            // interestingly, it dn save on resources 
            // when PTX compiled to SASS, the SMEM loads from Bs are vectorized!
            float tempB = Bs[dotIdx * BN + threadCol];

            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tempB;
            }
        }

        // sync before writing to C
        __syncthreads();
    }

    // write results to C
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] 
                    + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}


/*

issue:
- kernel still suffers from the same memory-bound problem

solution for this issue in the next kernel:
- get each thread to compute more for higher arithmetic intensity
- can do it using 2D blocktiling instead of 1D

*/
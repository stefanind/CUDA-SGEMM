#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
        sgemm_2D_blocktiling(int M, int N, int K, float alpha, const float *A, 
                              const float *B, float beta, float *C) {


    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    // one block requires numThreadsBlocktile amount of threads
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // num of threads must equal blockDim.x (threads assigned per block)
    assert(numThreadsBlocktile == blockDim.x);


    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    // SMEM assignment
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move starting positions
    A += cRow * K * BM; 
    B += cCol * BN;
    C += cRow * N * BM + cCol * BN;

    // for traversing A and B elements
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    // storage of preliminary results for each thread
    float threadResults[TM * TN] = {0.0};

    // for storing values of As and Bs in registers 
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        // GMEM to SMEM loads
        // each thread loads one element from a stripe
        // iterates to the next stripe so that the respected thread loads another  
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = 
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // iterate over SMEM (As and Bs)
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // SMEM to register loads
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }

            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            // register loads/writes 
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN]; 
                }
            }
        }

        __syncthreads();

    }

    // write to GMEM
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] + 
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }


    /*

    issue:
    - memory pipeline congestion, i.e., "Stall MIO Throttle" 
    - the warp cannot issue the next memory instruction because pipeline is full
    
    solution for the next kernel:
    - vectorize loads from As
    https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

    */ 
}
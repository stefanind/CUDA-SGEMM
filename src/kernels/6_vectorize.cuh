#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


template <const int BM, const int BN, const int BK, const int TM, const int TN>

__global__ void sgemm_vectorize(int M, int N, int K, float alpha, const float *A, 
                              const float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    // SMEM assignment
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * K * BM; 
    B += cCol * BN;
    C += cRow * N * BM + cCol * BN;

    // for vectorization, want to rely on LDS.128 in SASS
    // therefore, partition tile by dividing by 4 
    // to vectorize the load of a tile 
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // storage of preliminary results for each thread
    float threadResults[TM * TN] = {0.0};

    // for storing values of As and Bs in registers 
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        // GMEM to SMEM loads
        float4 temp = 
            // temp becomes the first 4 elements from the index
            reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        // load the 4 into SMEM using transpose of A
        As[(innerColA * 4 + 0) * BM + innerRowA] = temp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = temp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = temp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = temp.w;

        // why is this faster than manually unrolling it?
        // if unrolled, why doesn't the compiler coalesce it to also give 128b loads?
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        // adv blocktile
        A += BK;
        B += BK * N;

        // iterate over SMEM 
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // SMEM to register loads
            for (uint i = 0; i < TM; ++i) {
                // transposed formula
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) { // skip by 4
            // load C into registers
            float4 temp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            // update values that will be written to C in registers
            temp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * temp.x;
            temp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * temp.y;
            temp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * temp.z;
            temp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * temp.w;
            
            // write to GMEM
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = temp;
        }
    }


    /*


    */ 
}
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>

__global__ void sgemm_bank_conflicts(int M, int N, int K, float alpha, float *A, 
                                    float *B, float beta, float *C) {

    const uint row = blockIdx.y;
    const uint col = blockIdx.x;

    const uint col_thread = threadIdx.x % (BN / TN);
    const uint row_thread = threadIdx.x / (BN / TN);

    __shared__ As[BM * BK];
    __shared__ Bs[BK * BN];

    A += row * K * BM;
    B += col * BN;
    C += row * K * BM + col * BN;

    const uint row_inner_A = threadIdx.x / (BK / 4);
    const uint col_inner_A = threadIdx.x % (BK / 4);
    const uint row_inner_B = threadIdx.x / (BN / 4);
    const uint col_inner_B = threadIdx.x % (BN / 4);

    float results[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // compile-time helpers for Bs layout
    // helpful to guarantee unrolling (#pragma unroll) 
    constexpr int TN_ = TN;
    constexpr int LD  = BN / TN_;  

    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK) {

        // populate SMEM buffers
        // transpose A when storing (i.e., col_inner becomes row_inner)
        float4 temp = 
            reinterpret_cast<float4*>(&A[row_inner_A * K + col_inner_A * 4])[0];
        As[((col_inner_A * 4 + 0) * BM + row_inner_A)] = temp.x;
        As[((col_inner_A * 4 + 1) * BM + row_inner_A)] = temp.y;
        As[((col_inner_A * 4 + 2) * BM + row_inner_A)] = temp.z;
        As[((col_inner_A * 4 + 3) * BM + row_inner_A)] = temp.w;

        // swizzle Bs for avoiding bank conflicts
        tempB = reinterpret_cast<float4*>(&B[row_inner_B * N + col_inner_B * 4])[0];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int n = col_inner_B * 4 + k;       // column index in the tile

            int col_thread_s = n / TN_;        // which "thread column" will use it
            int i_s          = n % TN_;        // which of TN elements within that thread

            int rowp        = row_inner_B * TN_ + i_s;
            int idx         = rowp * LD + col_thread_s;

            float val = reinterpret_cast<float*>(&tempB)[k];
            Bs[idx] = val;
        }

        A += BK;
        B += BK * N;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {

            for (uint i = 0; i < TM; ++i) {
                // ensure transpose indexing
                regM[i] = As[dot_idx * BM + row_thread * TM  + i];
            }

            for (uint i = 0; i < TN; ++i) {
                // ensure swizzle indexing
                regN[i] = Bs[(dot_idx * TN_ + i) * LD + col_thread]
            }

            for (uint resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
                for (uint resIdx_N = 0; resIdx_N < TN; ++resIdx_N) {
                    results[resIdx_M * TN + resIdx_N] +=  
                        regM[resIdx_M] * regN[resIdx_N];
                }
            }
        }

        __syncthreads();
    }

    for (uint resIdx_M = 0; resIdx_M < TM; resIdx_M += 1) {
        for (uint resIdx_N = 0; resIdx_N < TN; resIdx_N += 4) {
            // load vectorized C into registers
            float temp = reinterpret_cast<float4*>(
                &C[(row_thread * TM + resIdx_M) * N + col_thread * TN + resIdx_N])[0];
            temp.x = alpha * results[resIdx_M * TN + resIdx_N + 0] + beta * temp.x;
            temp.y = alpha * results[resIdx_M * TN + resIdx_N + 1] + beta * temp.y;
            temp.z = alpha * results[resIdx_M * TN + resIdx_N + 2] + beta * temp.z;
            temp.w = alpha * results[resIdx_M * TN + resIdx_N + 3] + beta * temp.w;

            reinterpret_cast<float4*>(
                &C[(row_thread * TM + resIdx_M) * N + col_thread * TN + resIdx_N])[0] = temp;
        }
    }
}
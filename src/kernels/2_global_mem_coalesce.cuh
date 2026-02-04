#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


/*

BLOCKSIZE = 32
gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1)
blockDim(BLOCKSIZE * BLOCKSIZE, 1, 1) -> changed so all threads are indexed along x

*/


// make BLOCKSIZE a constant known at compile time
template <const uint BLOCKSIZE>

__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
    /*
    CHANGES:
    assign threads to memory more efficiently
    when row = 0, col = 0,1,...,BLOCKSIZE-1 for a whole warp
    therefore, for temp += A[0*K+i] * B[i * N + y_t] where y_t = 0,1,...,BLOCKSIZE-1,
    it means that a warp grabs from GMEM efficiently via coalescing,
    i.e., 32 threads grabs the 32 contiguous memory addresses associated with B 
    for each iteration of i. 
    
    So the whole matrix of B is retrieved contiguously thus increasing efficiency

    */
    const uint row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < M && col < N) {
        float temp = 0.0;
        for (int i = 0; i < K; ++i) {
            temp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
    }
}

/*

issue:
- does not use shared memory; it calls everything from global memory.

solution for this issue in next kernel:
- store As and Bs in shared memory to cache data that will be re-used.

*/
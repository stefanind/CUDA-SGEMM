// src/kernels.cuh
#pragma once

#include <cuda_runtime.h>

// include whichever kernels wanted
#include "kernels/1_naive.cuh"
#include "kernels/2_global_mem_coalesce.cuh"
#include "kernels/3_shared_mem_blocking.cuh"
#include "kernels/4_1D_blocktiling.cuh"
#include "kernels/5_2D_blocktiling.cuh"
#include "kernels/6_vectorize.cuh"
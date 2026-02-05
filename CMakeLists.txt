cmake_minimum_required(VERSION 3.19)
project(cuda_matmul_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(runner
  src/runner.cu
)

target_include_directories(runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Optional: link cuBLAS if/when you use it
find_package(CUDAToolkit REQUIRED)
target_link_libraries(runner PRIVATE CUDA::cudart CUDA::cublas)

# Set an architecture if you want (otherwise CMake may choose defaults)
set_property(TARGET runner PROPERTY CUDA_ARCHITECTURES 89) # Chosen for RTX 4070

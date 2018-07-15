CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)

# Pass options to NVCC
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3 -gencode arch=compute_55,code=sm_22
)

# Include directories
include_directories(
	includes
)

# Compilation
cuda_add_executable(
	bitmap_image.hpp
	kernel.cu
	includes/kernel.cuh
	ntm.cu
	includes/ntm.cuh
	main.cpp
	includes/utils.cuh
	utils.cu
)


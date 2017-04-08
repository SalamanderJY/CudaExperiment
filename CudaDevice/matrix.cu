#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.hpp>

#include <stdlib.h>
#include <random>
#include <iostream>

#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	// Loop over the d_M and d_N tiles required to compute d_P element.
	for (int m = 0; m < Width / TILE_WIDTH; ++m)
	{
		// Coolaborative loading of d_M and d_N tiles into shared memory
		Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	d_P[Row * Width + Col] = Pvalue;
}

int matrix_main(int argc, char* argv[])
{
	float* matrix_A = new float[1024 * 1024];
	float* matrix_B = new float[1024 * 1024];
	float* result = new float[1024 * 1024];

	float* dev_A = 0;
	float* dev_B = 0;
	float* dev_result = 0;

	cudaMalloc(&dev_A, 1024 * 1024 * sizeof(float));
	cudaMalloc(&dev_B, 1024 * 1024 * sizeof(float));
	cudaMalloc(&dev_result, 1024 * 1024 * sizeof(float));

	delete matrix_A;
	delete matrix_B;

}
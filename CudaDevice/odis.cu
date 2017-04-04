#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.hpp>

#include <stdlib.h>
#include <random>
#include <iostream>

#define TPB 1024
#define ATOMIC 1

#define COLS 10000
#define ROWS 1024

typedef int DATATYPE;

__global__ void OdisKernel(DATATYPE *d_res, const DATATYPE *d_a, const DATATYPE *d_b, int n)
{
	
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n) return;
	const int s_idx = threadIdx.x;

	__shared__ DATATYPE s_prod[TPB];
	s_prod[s_idx] = (d_a[idx] - d_b[idx]) * (d_a[idx] - d_b[idx]);
	__syncthreads();

	if (s_idx == 0)
	{
		int blockSum = 0;
		for (int j = 0; j < blockDim.x; ++j)
		{
			blockSum += s_prod[j];
		}

		//printf("Block_%d, blockSum = %d\n", blockIdx.x, blockSum);

		if (ATOMIC)
		{
			atomicAdd(d_res, blockSum);
		}
		else
		{
			*d_res += blockSum;
		}
	}
	
	
}

void odistance(DATATYPE *res, const DATATYPE *a, const DATATYPE *b, int n)
{
	DATATYPE *d_res = 0;
	DATATYPE *d_a = 0;
	DATATYPE *d_b = 0;

	cudaMalloc(&d_res, sizeof(DATATYPE));
	cudaMalloc(&d_a, n*sizeof(DATATYPE));
	cudaMalloc(&d_b, n*sizeof(DATATYPE));

	cudaMemset(d_res, 0, sizeof(DATATYPE));
	cudaMemcpy(d_a, a, n*sizeof(DATATYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n*sizeof(DATATYPE), cudaMemcpyHostToDevice);

	OdisKernel << <(n + TPB - 1) / TPB, TPB >> >(d_res, d_a, d_b, n);
	cudaMemcpy(res, d_res, sizeof(DATATYPE), cudaMemcpyDeviceToHost);

	cudaFree(d_res);
	cudaFree(d_a);
	cudaFree(d_b);
}

int odis_main(int argc, char* argv[])
{
	DATATYPE cpu_res = 0;
	DATATYPE gpu_res = 0;
	DATATYPE **a = new DATATYPE*[ROWS];
	for (int i = 0; i < ROWS; i++)
		a[i] = new DATATYPE[COLS];

	std::random_device rd;
    std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 10);

	for (int i = 0; i < ROWS * COLS; ++i)
	{
		a[i / COLS][i % COLS] = (DATATYPE)dis(gen);
	}

	int count = 0;
	
	for (int i = 0; i < ROWS; ++i)
	{
		for (int j = i + 1; j < ROWS; ++j)
		{
			cpu_res = 0;
			for (int k = 0; k < COLS; ++k)
			  cpu_res += (a[i][k] - a[j][k]) * (a[i][k] - a[j][k]);
			cpu_res = (DATATYPE)sqrt(cpu_res);
			if (count % 100000 == 0)
			  printf("%d : cpu result = %d\n", count, cpu_res);
			count++;
		}
	}
	
	count = 0;
	for (int i = 0; i < ROWS; ++i)
	{
		for (int j = i + 1; j < ROWS; ++j)
		{
			odistance(&gpu_res, a[i], a[j], COLS);
			if (count % 10000 == 0)
				printf("%d : gpu result = %d\n", count, (DATATYPE)sqrt(gpu_res));
			count++;
		}
	}
	
	
	system("pause");

	return 0;
}
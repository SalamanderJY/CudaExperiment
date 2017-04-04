#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.hpp>

#include <stdlib.h>

#define TPB 64
#define ATOMIC 1

#define N 1024

__global__ void dotKernel(int *d_res, const int *d_a, const int *d_b, int n)
{
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n) return;
	const int s_idx = threadIdx.x;

	__shared__ int s_prod[TPB];
	s_prod[s_idx] = d_a[idx] * d_b[idx];
	__syncthreads();

	if (s_idx == 0)
	{
		int blockSum = 0;
		for (int j = 0; j < blockDim.x; ++j)
		{
			blockSum += s_prod[j];
		}

		printf("Block_%d, blockSum = %d\n", blockIdx.x, blockSum);

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

void dotLauncher(int *res, const int *a, const int *b, int n)
{
	int *d_res;
	int *d_a = 0;
	int *d_b = 0;

	cudaMalloc(&d_res, sizeof(int));
	cudaMalloc(&d_a, n*sizeof(int));
	cudaMalloc(&d_b, n*sizeof(int));

	cudaMemset(d_res, 0, sizeof(int));
	cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

	dotKernel << <(n + TPB - 1) / TPB, TPB >> >(d_res, d_a, d_b, n);
	cudaMemcpy(res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_res);
	cudaFree(d_a);
	cudaFree(d_b);
}

int atomic_main(int argc, char* argv[])
{
	int cpu_res = 0;
	int gpu_res = 0;
	int *a = (int*)malloc(N*sizeof(int));
	int *b = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i)
	{
		a[i] = 1;
		b[i] = 1;
	}

	for (int i = 0; i < N; ++i)
	{
		cpu_res += a[i] * b[i];
	}
	printf("cpu result = %d\n", cpu_res);

	dotLauncher(&gpu_res, a, b, N);
	printf("gpu_result = %d\n", gpu_res);

	free(a);
	free(b);

	system("pause");

	return 0;
}
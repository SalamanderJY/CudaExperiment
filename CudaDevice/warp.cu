#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void GetThreadId(
	unsigned int *block, 
	unsigned int *thread, 
	unsigned int *warp, 
	unsigned int *calc_thread, 
	unsigned int *clocks)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	block[thread_id] = blockIdx.x;
	thread[thread_id] = threadIdx.x;
	warp[thread_id] = threadIdx.x / warpSize;
	calc_thread[thread_id] = thread_id;
	clocks[thread_id] = clock();
}
#define ArraySize Arraysize * sizeof(unsigned int)

int warp_main(int argc, char* argv[])
{
	unsigned int block_nums, thread_nums;
	printf("Input blocknums and threadnums\n");
	scanf("%d%d", &block_nums, &thread_nums);
	const unsigned int Arraysize = block_nums * thread_nums;

	unsigned int *gpu_block, *gpu_thread, *gpu_warp, *gpu_calc, *gpu_clock;

	cudaMalloc((void**)&gpu_block, ArraySize);
	cudaMalloc((void**)&gpu_thread, ArraySize);
	cudaMalloc((void**)&gpu_warp, ArraySize);
	cudaMalloc((void**)&gpu_calc, ArraySize);
	cudaMalloc((void**)&gpu_clock, ArraySize);

	GetThreadId << <block_nums, thread_nums >> >(gpu_block, gpu_thread, gpu_warp, gpu_calc, gpu_clock);

	unsigned int* cpu_block = new unsigned int[Arraysize];
	unsigned int* cpu_thread = new unsigned int[Arraysize];
	unsigned int* cpu_warp = new unsigned int[Arraysize];
	unsigned int* cpu_calc = new unsigned int[Arraysize];
	unsigned int* cpu_clock = new unsigned int[Arraysize];

	cudaMemcpy(cpu_block, gpu_block, ArraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ArraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_warp, gpu_warp, ArraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc, gpu_calc, ArraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_clock, gpu_clock, ArraySize, cudaMemcpyDeviceToHost);

	cudaFree(gpu_block); 
	cudaFree(gpu_thread);
	cudaFree(gpu_warp); 
	cudaFree(gpu_calc); 
	cudaFree(gpu_clock);

	for (int i = 0; i<Arraysize; i++)
	{
		// output the paramters, ps:cpu_clock[i] - cpu_clock[0] is represented of the difference between the run time of each thread and start thread,it is easy to calculate and observe.
		printf("Calculated Thread: %3u- Block: %3u- Warp: %3u- Thread: %3u- Time: %3u\n", cpu_calc[i], cpu_block[i], cpu_warp[i], cpu_thread[i], cpu_clock[i] - cpu_clock[0]);
	}

	delete cpu_block;
	delete cpu_thread;
	delete cpu_warp;
	delete cpu_calc;
	delete cpu_clock;

	system("pause");
	return 0;
}


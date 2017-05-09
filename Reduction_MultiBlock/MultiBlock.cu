
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "cutil_inline.h"

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////


void reduction_gold(float* odata, float* idata, const unsigned int len)
{
	*odata = 0;
	for (int i = 0; i<len; i++) *odata += idata[i];
	//printf("result:%f\n",*odata);
}


////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
	// dynamically allocated shared memory

	extern  __shared__  float temp[];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int k = threadIdx.x;

	// first, each thread loads data into shared memory

	temp[k] = g_idata[tid];

	// next, we perform binary tree reduction
	int d = blockDim.x; if (d % 2) temp[0] += temp[d - 1];
	for (d >>= 1; d > 0; d >>= 1) {
		__syncthreads();  // ensure previous step completed 
		if (k<d) { temp[k] += temp[k + d]; }
		if (k == 0 && d % 2 == 1 && d != 1) { temp[0] += temp[d - 1]; }
		//printf("middle result:d:%d  temp[%d]:%f\n",d,k,temp[k]);
	}

	// finally, first thread puts result into global memory

	if (tid == blockIdx.x*blockDim.x) {
		g_odata[blockIdx.x] = temp[0];
		//printf("g[%d]:%f\n",blockIdx.x,g_odata[blockIdx.x]);
	}
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	int num_elements, num_threads, mem_size, shared_mem_size, num_block;

	float *h_data, *reference, sum;
	float *d_idata, *d_odata;

	cutilDeviceInit(argc, argv);

	num_elements = 512; num_block = 2;
	num_threads = num_elements / num_block;
	mem_size = sizeof(float)* num_elements;

	// allocate host memory to store the input data
	// and initialize to integer values between 0 and 1000

	h_data = (float*)malloc(mem_size);

	for (int i = 0; i < num_elements; i++)
	{
		h_data[i] = floorf(1000 * (rand() / (float)RAND_MAX));
	}

	// compute reference solutions

	reference = (float*)malloc(mem_size);
	reduction_gold(&sum, h_data, num_elements);

	// allocate device memory input and output arrays

	cudaSafeCall(cudaMalloc((void**)&d_idata, mem_size));
	cudaSafeCall(cudaMalloc((void**)&d_odata, num_block*sizeof(float)));

	// copy host memory to device input array

	cudaSafeCall(cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

	// execute the kernel

	shared_mem_size = sizeof(float)* (num_elements / 2);
	reduction << <num_block, num_threads, shared_mem_size >> >(d_odata, d_idata);
	cudaCheckMsg("reduction kernel execution failed");

	// copy result from device to host


	if (num_block>1)
		/*for(int i=1;i<num_block;i++)
		{ h_data[0]+=h_data[i];
		//printf("GPU: h[%d]:%f\n",i,h_data[i]);
		}/*/
		reduction << <1, num_block, shared_mem_size >> >(d_odata, d_odata);

	cudaSafeCall(cudaMemcpy(h_data, d_odata, num_block*sizeof(float),
		cudaMemcpyDeviceToHost));

	// check results

	printf("reduction error = %f\n", h_data[0] - sum);

	// cleanup memory

	free(h_data);
	free(reference);
	cudaSafeCall(cudaFree(d_idata));
	cudaSafeCall(cudaFree(d_odata));

	// CUDA exit -- needed to flush printf write buffer

	cudaDeviceReset();
}

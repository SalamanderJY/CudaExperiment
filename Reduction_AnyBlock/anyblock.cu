
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
  for(int i=0; i<len; i++) *odata += idata[i]; 
}


////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid];

    // next, we perform binary tree reduction
	int d=blockDim.x; 
	if(d%2) temp[0]+=temp[d-1];

    for (d>>=1; d > 0; d >>= 1) { 
      __syncthreads();  // ensure previous step completed 

	if (tid<d) 
	{ temp[tid] += temp[tid+d]; }

	if(tid==0 && d%2==1 && d!=1) { temp[0]+=temp[d-1]; }

    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[0] = temp[0];
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv) 
{
  int num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, *reference, sum;
  float *d_idata, *d_odata;

  cutilDeviceInit(argc, argv);

  num_elements =369;
  num_threads  = num_elements;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
   { h_data[i] = floorf(1000*(rand()/(float)RAND_MAX)); }

  // compute reference solutions

  reference = (float*) malloc(mem_size);
  reduction_gold(&sum, h_data, num_elements);

  // allocate device memory input and output arrays

  cudaSafeCall(cudaMalloc((void**)&d_idata, mem_size));
  cudaSafeCall(cudaMalloc((void**)&d_odata, sizeof(float)));

  // copy host memory to device input array

  cudaSafeCall(cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * num_elements;
  reduction<<<1,num_threads,shared_mem_size>>>(d_odata,d_idata);
  cudaCheckMsg("reduction kernel execution failed");

  // copy result from device to host

  cudaSafeCall(cudaMemcpy(h_data, d_odata, sizeof(float),
                           cudaMemcpyDeviceToHost));

  // check results

  printf("reduction error = %f\n",h_data[0]-sum);

  // cleanup memory

  free(h_data);
  free(reference);
  cudaSafeCall(cudaFree(d_idata));
  cudaSafeCall(cudaFree(d_odata));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cutil_inline.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>

// CUDA global constant variables
__constant__ int N;
__constant__ float a, b, c;

// Kernel Function
__global__ void Polynomial(float *d_z, float *d_v)
{
	float y1, sum = 0.0f;

	// version1
	//d_z = d_z + threadIdx.x + 2 * N * blockIdx.x * blockDim.x;

	// version2
	d_z = d_z + 2 * N * threadIdx.x + 2 * N * blockIdx.x * blockDim.x;

	d_v = d_v + threadIdx.x + blockIdx.x * blockDim.x;

	for (int n = 0; n < N; n++)
	{
		y1 = (*d_z);
		// version1
		//d_z += blockDim.x;
		// version2
		d_z += 1;

		sum += a * y1 * y1 + b * y1 + c;
	}

	*d_v = sum / N;
}

int main_poly(int argc, char* argv[]) {

	int     NPATH = 960000, h_N = 100;
	float   h_a, h_b, h_c;
	float  *h_v, *d_v, *d_z;
	double  sum1, sum2;

	//double timer, elapsed;  
	clock_t timer;   // for counting the CPU time
	double elapsed;	 // elapsed time	

	curandGenerator_t gen;

	// initialise card

	cutilDeviceInit(argc, argv);

	// allocate memory on host and device

	h_v = (float *)malloc(sizeof(float)*NPATH);

	cudaSafeCall(cudaMalloc((void **)&d_v, sizeof(float)*NPATH));
	cudaSafeCall(cudaMalloc((void **)&d_z, sizeof(float)* 2 * h_N * NPATH));

	// define constants and transfer to GPU

	h_a = 1.0f;
	h_b = 2.0f;
	h_c = 0.0f;

	cudaSafeCall(cudaMemcpyToSymbol(N, &h_N, sizeof(h_N)));
	cudaSafeCall(cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)));
	cudaSafeCall(cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)));
	cudaSafeCall(cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)));

	// random number generation

	timer = clock();  // initialise timer

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateNormal(gen, d_z, 2 * h_N * NPATH, 0.0f, 1.0f);

	cudaSafeCall(cudaDeviceSynchronize());

	elapsed = elapsed_time(&timer);
	printf("\nCURAND normal RNG execution time (ms): %f ,   samples/sec: %e \n",
		elapsed, 2.0*h_N*NPATH / elapsed);

	// execute kernel and time it

	Polynomial << <NPATH / 64, 64 >> >(d_z, d_v);
	cudaCheckMsg("pathcalc execution failed\n");
	cudaSafeCall(cudaDeviceSynchronize());

	elapsed = elapsed_time(&timer);
	printf("Polynomial kernel execution time (ms): %f \n", elapsed);

	// copy back results

	cudaSafeCall(cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
		cudaMemcpyDeviceToHost));

	// compute average

	sum1 = 0.0;
	sum2 = 0.0;
	for (int i = 0; i < NPATH; i++) {
		sum1 += h_v[i];
		//printf("%f\n", h_v[i]);
		sum2 += h_v[i] * h_v[i];
	}
	
	printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
		sum1 / NPATH, sqrt((sum2 / NPATH - (sum1 / NPATH)*(sum1 / NPATH)) / NPATH));

	// Tidy up library

	curandDestroyGenerator(gen);

	// Release memory and exit cleanly

	free(h_v);
	cudaSafeCall(cudaFree(d_v));
	cudaSafeCall(cudaFree(d_z));

	// CUDA exit -- needed to flush printf write buffer

	cudaDeviceReset();
	system("pause");

	return 0;
}

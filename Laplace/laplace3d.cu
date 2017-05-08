//
// Program to solve Laplace equation on a regular 3D grid
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cutil_inline.h"
#include <stdio.h>

////////////////////////////////////////////////////////////////////////
// include kernel function
///////////////////////////////////////////////////////////////////////
//
// Notes:one thread per node in the 2D block;
// after initialisation it marches in the k-direction
//
////////////////////////////////////////////////////////////////////////
// define kernel block size for 
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 32
#define BLOCK_Y 6


__global__ void GPU_laplace3d(int NX, int NY, int NZ, float *d_u1, float *d_u2)
{
	int   i, j, k, indg, active, IOFF, JOFF, KOFF;
	float u2, sixth = 1.0f / 6.0f;

	//
	// define global indices and array offsets
	//

	i = threadIdx.x + blockIdx.x*BLOCK_X;
	j = threadIdx.y + blockIdx.y*BLOCK_Y;
	indg = i + j*NX;

	IOFF = 1;
	JOFF = NX;
	KOFF = NX*NY;

	active = i >= 0 && i <= NX - 1 && j >= 0 && j <= NY - 1;

	for (k = 0; k<NZ; k++) {

		if (active) {
			if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
				u2 = d_u1[indg];  // Dirichlet b.c.'s
			}
			else {
				u2 = (d_u1[indg - IOFF] + d_u1[indg + IOFF]
					+ d_u1[indg - JOFF] + d_u1[indg + JOFF]
					+ d_u1[indg - KOFF] + d_u1[indg + KOFF]) * sixth;
			}
			d_u2[indg] = u2;

			indg += KOFF;
		}
	}
}


////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////
void Gold_laplace3d(int NX, int NY, int NZ, float* h_u1, float* h_u2);

clock_t elapsed_time(clock_t *timer)
{
	return clock() - *timer;
}
////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

	// 'h_' prefix - CPU (host) memory space

	int    NX = 200, NY = 200, NZ = 200, REPEAT = 100,
		bx, by, i, j, k, ind;
	float  *h_u1, *h_u2, *h_u3, *h_foo, err;

	clock_t timer;
	double elapsed;

	// 'd_' prefix - GPU (device) memory space

	float  *d_u1, *d_u2, *d_foo;

	printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

	// initialise card

	cutilDeviceInit(argc, argv);

	// allocate memory for arrays

	h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_u3 = (float *)malloc(sizeof(float)*NX*NY*NZ);
	cudaSafeCall(cudaMalloc((void **)&d_u1, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_u2, sizeof(float)*NX*NY*NZ));

	// initialise u1

	for (k = 0; k<NZ; k++) {
		for (j = 0; j<NY; j++) {
			for (i = 0; i<NX; i++) {
				ind = i + j*NX + k*NX*NY;

				if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
					h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
				else
					h_u1[ind] = 0.0f;
			}
		}
	}

	// copy u1 to device

	timer = clock();
	cudaSafeCall(cudaMemcpy(d_u1, h_u1, sizeof(float)*NX*NY*NZ,
		cudaMemcpyHostToDevice));
	elapsed = elapsed_time(&timer);
	printf("\nCopy u1 to device: %f (ms) \n", elapsed);

	// Set up the execution configuration

	bx = 1 + (NX - 1) / BLOCK_X;
	by = 1 + (NY - 1) / BLOCK_Y;

	dim3 dimGrid(bx, by);
	dim3 dimBlock(BLOCK_X, BLOCK_Y);

	// printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
	// printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

	// Execute GPU kernel

	for (i = 1; i <= REPEAT; ++i) {
		GPU_laplace3d << <dimGrid, dimBlock >> >(NX, NY, NZ, d_u1, d_u2);
		cudaCheckMsg("GPU_laplace3d execution failed\n");

		d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u2
	}

	cudaSafeCall(cudaDeviceSynchronize());
	elapsed = elapsed_time(&timer);
	printf("\n%dx GPU_laplace3d_naive: %f (ms) \n", REPEAT, elapsed);

	// Read back GPU results

	cudaSafeCall(cudaMemcpy(h_u2, d_u1, sizeof(float)*NX*NY*NZ,
		cudaMemcpyDeviceToHost));
	elapsed = elapsed_time(&timer);
	printf("\nCopy u2 to host: %f (s) \n", elapsed);

	// print out corner of array

	/*
	for (k=0; k<3; k++) {
	for (j=0; j<8; j++) {
	for (i=0; i<8; i++) {
	ind = i + j*NX + k*NX*NY;
	printf(" %5.2f ", h_u2[ind]);
	}
	printf("\n");
	}
	printf("\n");
	}
	*/

	// Gold treatment

	for (int i = 1; i <= REPEAT; ++i) {
		Gold_laplace3d(NX, NY, NZ, h_u1, h_u3);
		h_foo = h_u1; h_u1 = h_u3; h_u3 = h_foo;   // swap h_u1 and h_u3
	}

	elapsed = elapsed_time(&timer);
	printf("\n%dx Gold_laplace3d: %f (ms) \n \n", REPEAT, elapsed);

	// print out corner of array

	/*
	for (k=0; k<3; k++) {
	for (j=0; j<8; j++) {
	for (i=0; i<8; i++) {
	ind = i + j*NX + k*NX*NY;
	printf(" %5.2f ", h_u1[ind]);
	}
	printf("\n");
	}
	printf("\n");
	}
	*/

	// error check

	err = 0.0;

	for (k = 0; k<NZ; k++) {
		for (j = 0; j<NY; j++) {
			for (i = 0; i<NX; i++) {
				ind = i + j*NX + k*NX*NY;
				err += (h_u1[ind] - h_u2[ind])*(h_u1[ind] - h_u2[ind]);
			}
		}
	}

	printf("\nrms error = %f \n", sqrt(err / (float)(NX*NY*NZ)));

	// Release GPU and CPU memory

	cudaSafeCall(cudaFree(d_u1));
	cudaSafeCall(cudaFree(d_u2));
	free(h_u1);
	free(h_u2);
	free(h_u3);

	cudaDeviceReset();

	system("pause");
}

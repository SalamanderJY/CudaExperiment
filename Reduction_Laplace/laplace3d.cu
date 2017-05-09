//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>




////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 64
#define BLOCK_Y 4

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include "laplace3d_kernel.cu"
#include "multiblock.cu"

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_laplace3d(int NX, int NY, int NZ, float* h_u1, float* h_u2);

////////////////////////////////////////////////////////////////////////
//time function
////////////////////////////////////////////////////////////////////////

double elapsed_time(double *timer)
{
	double t1,t2;   t2=(double)clock()/CLOCKS_PER_SEC;
	t1= t2-*timer;  *timer=t2;
	return t1;
}

//////////////////////////////////////////////////////////////////////////

__global__ void dissq(float* u1,float* u2,long int *ad)
{
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  u1[tid+*ad]=(u1[tid+*ad]-u2[tid+*ad])*(u1[tid+*ad]-u2[tid+*ad]);
 // printf("thread%d: %f",tid+*ad,u1[tid+*ad]);
}


//////////////////////////////////////////////////////////////////////////


void checkCudaErr(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
////////////////////////////////////////////////////////////////////////////////
void Gold_laplace3d(int NX, int NY, int NZ, float* u1, float* u2) 
{
  int   i, j, k, ind;
  float sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
                    + u1[ind-NX   ] + u1[ind+NX   ]
                    + u1[ind-NX*NY] + u1[ind+NX*NY] ) * sixth;
        }
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){




  // 'h_' prefix - CPU (host) memory space

  int    NX=200, NY=200, NZ=200, REPEAT=100,
         bx, by, i, j, k, ind;
  float  *h_u1, *h_u2, *h_u3, *h_foo;

  double timer, elapsed;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u1, *d_u2, *d_foo;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card

  cutilDeviceInit(argc, argv);
 
  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u3 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  cudaSafeCall(cudaMalloc((void **)&d_u1, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_u2, sizeof(float)*NX*NY*NZ) );

  // initialise u1
    
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  elapsed_time(&timer);  
  cudaSafeCall(cudaMemcpy(d_u1, h_u1, sizeof(float)*NX*NY*NZ,
                           cudaMemcpyHostToDevice) );
  elapsed = elapsed_time(&timer);
  printf("\nCopy u1 to device: %f (s) \n", elapsed);					;

  // Set up the execution configuration

  bx = 1 + (NX-1)/BLOCK_X;
  by = 1 + (NY-1)/BLOCK_Y;

  dim3 dimGrid(bx,by);
  dim3 dimBlock(BLOCK_X,BLOCK_Y);

  // printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
  // printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

  // Execute GPU kernel

  for (i = 1; i <= REPEAT; ++i) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    cudaCheckMsg("GPU_laplace3d execution failed\n");

    d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u2
  }

  cudaSafeCall(cudaDeviceSynchronize());
  elapsed = elapsed_time(&timer);
  printf("\n%dx GPU_laplace3d_naive: %f (s) \n", REPEAT, elapsed);			

  // Read back GPU results

  cudaSafeCall(cudaMemcpy(h_u2, d_u1, sizeof(float)*NX*NY*NZ,
                           cudaMemcpyDeviceToHost) );
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
  printf("\n%dx Gold_laplace3d: %f (s) \n \n", REPEAT, elapsed);			

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

 
/*
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;
        err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
      }
    }
  }
//*/
  cudaSafeCall(cudaMemcpy(d_u2, h_u1, sizeof(float)*NX*NY*NZ,
                           cudaMemcpyHostToDevice) );
  int num_block=800,num_thread=1000,mem_shared=num_thread*sizeof(float);
  long int *ad=0;
  cudaSafeCall(cudaMalloc((void**)&ad,sizeof(long int)) );
  long int *h_ad=(long int*)malloc(sizeof(long int));

  float *err_g;  cudaSafeCall( cudaMalloc((void**)&err_g,num_block*sizeof(float)) );
  float *err_h;  err_h=(float*)malloc(sizeof(float)*num_block); float err=0;
  for(*h_ad=0;*h_ad<NX*NY*NZ;*h_ad+=num_block*num_thread)
  { 
    cudaSafeCall( cudaMemcpy(ad,h_ad,sizeof(long int),cudaMemcpyHostToDevice) );
    dissq<<<num_block,num_thread>>>(d_u1,d_u2,ad);
    cudaThreadSynchronize();
    reduction<<<num_block,num_thread,mem_shared>>>(err_g,d_u1,ad);
    
   // reduction<<<1,num_block,num_block*sizeof(float)>>>(err_g,err_g,0);	checkCudaErr("adds error");
    cudaSafeCall( cudaMemcpy(err_h,err_g,sizeof(float)*num_block,
                           cudaMemcpyDeviceToHost) );
    for(int f=0;f<num_block;f++) err_h[0]+=err_h[f];
    err+=err_h[0];
  }
/*   
//*/
	
  printf("\n rms error = %f \n",sqrt(err/ (float)(NX*NY*NZ)));


 // Release GPU and CPU memory

  cudaSafeCall(cudaFree(d_u1)); cudaSafeCall(cudaFree(err_g));
  cudaSafeCall(cudaFree(d_u2)); 
  free(h_u1);
  free(h_u2);
  free(h_u3);
  //free(err_h);

  cudaDeviceReset();

return 0;

}

#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdio.h>
#include<cuda_runtime.h>

#include"cutil_inline.h"

//#include <Windows.h>

#define maxnum 10000
#define Arraysizeby (maxnum*anum)*sizeof(int)
#define Resizeby 10*sizeof(float)
const int anum = 1024;
int avector[anum][maxnum];
__global__ void Odis(int *o1, int *o2, float *result, int *cx, int *cy)
{
	extern __shared__ float temp[];
	int tid = threadIdx.x + (blockIdx.x*blockDim.x);
	int k = threadIdx.x;
	temp[k] = (float)(o1[tid + *cx] - o2[tid + *cy])*(o1[tid + *cx] - o2[tid + *cy]); //printf("temp[%d]:%f\n",k,temp[k]);
	__syncthreads();

	int d = blockDim.x; if (d % 2) temp[0] += temp[d - 1];
	for (d >>= 1; d > 0; d >>= 1) {
		__syncthreads();
		if (k<d) { temp[k] += temp[k + d]; }
		if (k == 0 && d % 2 == 1 && d != 1) { temp[0] += temp[d - 1]; }
		//printf("middle result:d:%d  temp[%d]:%f\n",d,k,temp[k]);
	}

	if (tid == blockIdx.x*blockDim.x) {
		result[blockIdx.x] = temp[0];
		//printf("g[%d]:%f\n",blockIdx.x,g_odata[blockIdx.x]);
	}

}

//const long int alens = anum*(anum - 1) / 2;
float result[Resizeby];

int main()
{
	int i = 0, j = 0;
	srand((int)time(0));
	for (i = 0; i<anum; i++)
	{
		//printf("Vector %d:\n",i);
		for (j = 0; j<maxnum; j++)
		{
			avector[i][j] = rand() % 100;
			//printf("%d ",avector[i][j]);
		}
		//printf("\n");
	}


	int *gpu_o1; float *gpu_result;
	cudaMalloc((void**)&gpu_o1, Arraysizeby);
	cudaMalloc((void**)&gpu_result, Resizeby*sizeof(float));

	int *cx, *cy;
	cudaSafeCall(cudaMalloc((void**)&cx, sizeof(int)));
	cudaSafeCall(cudaMalloc((void**)&cy, sizeof(int)));

	cudaMemcpy(gpu_o1, avector, Arraysizeby, cudaMemcpyHostToDevice);

	for (i = 0; i<Resizeby; i++) result[i] = 0;
	cudaMemcpy(gpu_result, result, Resizeby*sizeof(float), cudaMemcpyHostToDevice);
	int k = 0;
	for (i = 0; i<1024; i++)
	for (j = i + 1; j<1024; j++)
	{
		cudaSafeCall(cudaMemcpy(cx, &i, sizeof(int), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemcpy(cy, &j, sizeof(int), cudaMemcpyHostToDevice));
		Odis << <10, 1000, 1000 * sizeof(float) >> >(gpu_o1, gpu_o1, gpu_result, cx, cy);
		cudaDeviceSynchronize();
		printf(cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(result, gpu_result, Resizeby, cudaMemcpyDeviceToHost);
		for (int k = 1; k<10; k++) result[0] += result[k];
		printf("line %d: %f\n", k++, sqrt(result[0]));
	}


	return 0;

}
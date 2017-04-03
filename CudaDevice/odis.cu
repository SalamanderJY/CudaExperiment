#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iostream>

//CUDA RunTime API
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define THREAD_NUM 1024

#define COLS 10000
#define ROWS 32

const int block_num = ROWS * (COLS + THREAD_NUM - 1) / THREAD_NUM;

//生成随机矩阵
void matrixGen(float* data, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			data[i * cols + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
		}
	}
}

__global__ static void odis(const float* data, float* result, int rows, int cols)
{
	//从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
	const int idx = blockDim.x * blockIdx.x *THREAD_NUM + threadIdx.x;
	const int row = idx / cols;
	const int column = idx % cols;

	int count = 0;

	//计算矩阵乘法
	if (row < rows && column < cols)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = i + 1; j < rows; j++)
			{
				float temp = 0;
				for (int k = 0; k < cols; k++)
				{
					temp += (data[i * cols + k] - data[j * cols + k]) * (data[i * cols + k] - data[j * cols + k]);
				}
				result[count] = (float)sqrt(temp);
				count++;
			}
		}
	}

}

int odis_main(int argc, char* argv[])
{

	//定义矩阵
	float *data;
	data = (float*)malloc(sizeof(float)* ROWS * COLS);

	float *result;
	int result_size = ((ROWS - 1) + 1) * (ROWS - 1) / 2;
	result = (float*)malloc(sizeof(float)* result_size);

	//设置随机数种子
	srand(0);
	//随机生成矩阵
	matrixGen(data, ROWS, COLS);

	/*把数据复制到显卡内存中*/
	float *cuda_data;
	float *cuda_result;

	//cudaMalloc 取得一块显卡内存 
	cudaMalloc((void**)&cuda_data, sizeof(float)* ROWS * COLS);
	cudaMalloc((void**)&cuda_result, sizeof(float)* result_size);

	//cudaMemcpy 将产生的矩阵复制到显卡内存中
	//cudaMemcpyHostToDevice - 从内存复制到显卡内存
	//cudaMemcpyDeviceToHost - 从显卡内存复制到内存
	cudaMemcpy(cuda_data, data, sizeof(float)* ROWS * COLS, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_result, result, sizeof(float)* result_size, cudaMemcpyHostToDevice);

	// 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
	odis << < block_num, THREAD_NUM, 0 >> >(cuda_data, cuda_result, ROWS, COLS);
	cudaDeviceSynchronize();
	/*把结果从显示芯片复制回主内存*/

	//cudaMemcpy 将结果从显存中复制回内存
	cudaMemcpy(result, cuda_result, sizeof(float)* result_size, cudaMemcpyDeviceToHost);

	//Free
	cudaFree(cuda_data);
	cudaFree(cuda_result);

	for (int i = 0; i < result_size; i++)
		std::cout << i << ": " <<result[i] << " " << std::endl;
    
	system("pause");

	return 0;
}
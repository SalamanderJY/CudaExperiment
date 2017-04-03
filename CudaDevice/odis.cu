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

//�����������
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
	//�� bid �� tid �������� thread Ӧ�ü���� row �� column
	const int idx = blockDim.x * blockIdx.x *THREAD_NUM + threadIdx.x;
	const int row = idx / cols;
	const int column = idx % cols;

	int count = 0;

	//�������˷�
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

	//�������
	float *data;
	data = (float*)malloc(sizeof(float)* ROWS * COLS);

	float *result;
	int result_size = ((ROWS - 1) + 1) * (ROWS - 1) / 2;
	result = (float*)malloc(sizeof(float)* result_size);

	//�������������
	srand(0);
	//������ɾ���
	matrixGen(data, ROWS, COLS);

	/*�����ݸ��Ƶ��Կ��ڴ���*/
	float *cuda_data;
	float *cuda_result;

	//cudaMalloc ȡ��һ���Կ��ڴ� 
	cudaMalloc((void**)&cuda_data, sizeof(float)* ROWS * COLS);
	cudaMalloc((void**)&cuda_result, sizeof(float)* result_size);

	//cudaMemcpy �������ľ����Ƶ��Կ��ڴ���
	//cudaMemcpyHostToDevice - ���ڴ渴�Ƶ��Կ��ڴ�
	//cudaMemcpyDeviceToHost - ���Կ��ڴ渴�Ƶ��ڴ�
	cudaMemcpy(cuda_data, data, sizeof(float)* ROWS * COLS, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_result, result, sizeof(float)* result_size, cudaMemcpyHostToDevice);

	// ��CUDA ��ִ�к��� �﷨����������<<<block ��Ŀ, thread ��Ŀ, shared memory ��С>>>(����...);
	odis << < block_num, THREAD_NUM, 0 >> >(cuda_data, cuda_result, ROWS, COLS);
	cudaDeviceSynchronize();
	/*�ѽ������ʾоƬ���ƻ����ڴ�*/

	//cudaMemcpy ��������Դ��и��ƻ��ڴ�
	cudaMemcpy(result, cuda_result, sizeof(float)* result_size, cudaMemcpyDeviceToHost);

	//Free
	cudaFree(cuda_data);
	cudaFree(cuda_result);

	for (int i = 0; i < result_size; i++)
		std::cout << i << ": " <<result[i] << " " << std::endl;
    
	system("pause");

	return 0;
}
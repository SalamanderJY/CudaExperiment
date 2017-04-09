#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

bool CUDA_INIT(void)
{
	// cudaGetDeviceCount(&device_count)
	int device_count;
	if (cudaGetDeviceCount(&device_count))
	{
		printf(" There is zero device beyond 1.0\n");
		return false;
	}
	else
		printf("There is %d device beyond 1.0\n", device_count);

	//cudaGetDeviceProperties(&device_prop, i)
	int i;
	for (i = 0; i < device_count; i++)
	{
		struct cudaDeviceProp device_prop;
		if (cudaGetDeviceProperties(&device_prop, i) == cudaSuccess)
		{
			printf("device properties is :\n"
				"\t ASCII string identifying device is %s\n"
				"\t Global memory available on device in bytes (4G) is %lld\n"
				"\t Shared memory available per block in bytes is %lld\n"
				"\t 32-bit registers available per block is %d\n"
				"\t Warp size in threads is %d\n"
				"\t Maximum pitch in bytes allowed by memory copies is %lld\n"
				"\t Maximum number of threads per block is %d\n"
				"\t Maximum size of each dimension of a block is %d X %d X %d\n"
				"\t Maximum size of each dimension of a grid is %d X %d X %d\n"
				"\t Constant memory available on device in bytes is %lld\n"
				"\t Major compute capability is %d ,Minor compute capability is %d\n"
				"\t Clock frequency in kilohertz is %d\n"
				"\t Alignment requirement for textures is %lld\n"
				"\t Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount is %d\n"
				"\t Number of multiprocessors on device is %d\n",
				device_prop.name,                                 /**< ASCII string identifying device */
				(unsigned long long)device_prop.totalGlobalMem,   /**< Global memory available on device in bytes (2G)*/ 
				(unsigned long long)device_prop.sharedMemPerBlock,/**< Shared memory available per block in bytes */
				device_prop.regsPerBlock,                         /**< 32-bit registers available per block */
				device_prop.warpSize,                             /**< Warp size in threads */   
				(unsigned long long)device_prop.memPitch,         /**< Maximum pitch in bytes allowed by memory copies */
				device_prop.maxThreadsPerBlock,                   /**< Maximum number of threads per block */
				device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2], /**< Maximum size of each dimension of a block */
				device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2],       /**< Maximum size of each dimension of a grid */
				(unsigned long long)device_prop.totalConstMem,    /**< Constant memory available on device in bytes */
				device_prop.major,                                /**< Major compute capability */
				device_prop.minor,                                /**< Minor compute capability */
				device_prop.clockRate,                            /**< Clock frequency in kilohertz */
				(unsigned long long)device_prop.textureAlignment, /**< Alignment requirement for textures */
				device_prop.deviceOverlap,                        /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
				device_prop.multiProcessorCount);                 /**< Number of multiprocessors on device */
			break;
		}
	}

	if (i == device_count)
	{
		printf("Get the propertites of device occurred error\n");
		return false;
	}

	if (cudaSetDevice(i) == cudaErrorInvalidDevice)
	{
		printf("Set Device occurred error\n");
		return false;
	}

	return true;
}

int parameters_main(int argc, char* argv[])
{
	if (CUDA_INIT() == true)
		printf("CUDA INIT SUCCESSED!\n");

	system("pause");
	return 0;
}

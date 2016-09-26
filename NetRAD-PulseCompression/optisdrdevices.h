#ifndef __OPTISDRDEVICES__
#define __OPTISDRDEVICES__
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Include Cuda run-time and inline Libraries %%%%%%%%%%%%%%%%%
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
//
using namespace std;
//
//
class OptiSDRDevices
{

public:
	OptiSDRDevices(){optidx=0;}
	void setCudaDevice(int cuda_device)
	{
		countCudaDevices();
		if(cuda_device<numDevices)
		{
			cudaGetDevice(&cuda_device); // Get Just the CUDA Device name
			cudaSetDevice(cuda_device);		
			printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_device,getDeviceName(cuda_device));
			used_devices[optidx] = cuda_device;
			optidx= optidx + 1;
		}
		else
		{
			printf("[SDR_DSL_ERROR]$ No Cuda Device [%d] available\n",cuda_device);
		}
	}
	void countCudaDevices()
	{
		cudaGetDeviceCount(&numDevices);
	}
	char *getDeviceName(int devid)
	{
		cudaGetDeviceProperties(&deviceProp,devid);
		return deviceProp.name;
	}
private:
	cudaDeviceProp deviceProp;
	int used_devices[25]; // Change this to a Bigger Number for Big Super Computers
	int numDevices;
	int optidx;
	//
	//%%%%%%%%%%%%%% otherwise use device with highest Gflops/s %%%%%%%%%%%%%%%%
	//
};
#endif

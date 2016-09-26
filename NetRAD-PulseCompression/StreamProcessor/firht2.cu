//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: Stream Processing Techniques for OptiSDR %%%%%%%%%%%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%% Inlcude some C Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Include Cuda run-time and inline Libraries %%%%%%%%%%%%%%%%%
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

using namespace std;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%% Including the Kernel: FIR Filter Calculator %%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% NetRAD Raw ADC Data Read %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void readNetRadSamples(string strFilename, unsigned int uiNSamples, vector<short> &vsSamples)
{
	//Read
	ifstream oIFS;
	oIFS.open(strFilename.c_str(), ifstream::binary);
	if(!oIFS.is_open())
	{	
		cout << "[SDR_DSL_INFO]$ Error unable to open file \"" << strFilename << "\"" << endl;
		oIFS.close();
		exit(1);
	}

	vsSamples.resize(uiNSamples);

	oIFS.read((char*)&vsSamples.front(), sizeof(short) * uiNSamples);

	if(oIFS.gcount() << sizeof(short) * uiNSamples && oIFS.eof())
	{
		cout << "[SDR_DSL_INFO]$ Warning: hit end of file after " << oIFS.gcount() / sizeof(short) << " samples. Output is shortened accordingly." << endl;
		vsSamples.resize(oIFS.gcount() / sizeof(short));
	}

	oIFS.close();

	//int iTemp;

	//Covert from unsigned to signed - Uncomment lines below
	/*for(unsigned int uiSampleNo = 0; uiSampleNo < uiNSamples; uiSampleNo++)
	{
		iTemp = *((unsigned short*)&vsSamples[uiSampleNo]);
		iTemp -= 8192;
		vsSamples[uiSampleNo] = iTemp;
	}*/
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Complex Multiplication Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void complexvector_multiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		outp[tid] = cuCmulf(_cx[tid],_cy[tid]);
}
//
// This Kernel Utilize a 1-D Grid to a 1-D data indexes (1-D BlockDim = 1024 threads)
// We can multiply 16 Million floating point values in Parallel
__global__ void complexvector_multiply1d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//
	if(index < N)
	{
		outp[index].x = (_cx[index].x*_cy[index].x) - (_cx[index].y*_cy[index].y);//cuCmulf(_cx[index],_cy[index]);
		outp[index].y = (_cx[index].x*_cy[index].y) + (_cx[index].y*_cy[index].x);
	}
}
// This Kernel Utilize a 2-D Grid flattened to a 1-D data indexes (1-D BlockDim = 1024 threads)
// We can multiply 68.747788288 Million floating point values in Parallel
__global__ void complexvector_multiply2d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	//
	int xsize = blockDim.x*gridDim.x; // X dimension total number of thread
	//
	int index = tidy*xsize + tidx; // Index through 2-D grid
	//
	if(index < N)
		outp[index] = cuCmulf(_cx[index],_cy[index]);
}
// This Kernel Utilize a 3-D Grid flattened to a 1-D data indexes
// We can multiply 129 Million floating point values in Parallel
__global__ void complexvector_multiply3d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	int tidz = threadIdx.z + blockIdx.z * blockDim.z;
	//
	int xsize = blockDim.x*gridDim.x; // X dimension total number of thread
	int zsize = xsize*blockDim.y*gridDim.y; // Entire 2-D grid numer of thread
	//
	int xyindex = tidy*xsize + tidx; // Index through 2-D grid
	int index = tidz*zsize + xyindex; // Index through entire 3-D grid
	//
	if(index < N)
		outp[index] = cuCmulf(_cx[index],_cy[index]);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Complex Conjugate Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void complexvector_conjugate(cuFloatComplex *_cx, cuFloatComplex *outp,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		//outp[tid].x= (_cx[tid].x*_cy[tid].x)-(_cx[tid].y*_cy[tid].y);
		//outp[tid].y= (_cx[tid].x*_cy[tid].y)+(_cy[tid].x*_cx[tid].y);
		outp[tid] = cuConjf(_cx[tid]);
		tid += blockDim.x * gridDim.x;
	}
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Complex Absolute Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void complexvector_abs(cuFloatComplex *_cx,float *outp,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		//outp[tid].x= (_cx[tid].x*_cy[tid].x)-(_cx[tid].y*_cy[tid].y);
		//outp[tid].y= (_cx[tid].x*_cy[tid].y)+(_cy[tid].x*_cx[tid].y);
		outp[tid] = cuCabsf(_cx[tid]);
		tid += blockDim.x * gridDim.x;
	}
}
//
__global__ void optisdrifftscale(cuFloatComplex *invec, cuFloatComplex *out, int fp, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		out[tid] = make_cuFloatComplex(cuCrealf(invec[tid])/(float)fp,cuCimagf(invec[tid])/(float)fp);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%% Some Function Prototypes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void getShortSamples(vector<short> &vSamples, unsigned int nSamples);
void getRealSamples(vector<float> &vSamples, unsigned int nSamples);
float* getRealFSamples(int dalen);
void getRandSamples(vector<float> &vSamples, unsigned int nSamples);
void getAscendingRandSamples(vector<float> &vSamples, unsigned int nSamples);
cuFloatComplex* getComplexSamples(int _dlen);
cuFloatComplex* getComplexRandomSamples(int _dlen);

cuFloatComplex* getComplexSamples(vector<short> &vSamples, int _dlen)
{
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	for(int i = 0; i< _dlen; i++)
	{
		cSamples[i] = make_cuFloatComplex(vSamples[i],0.0);
	}
	return cSamples;
}
//
cuFloatComplex* getComplexSamples(vector<short> &vSamples, int from, int to)
{
	int dlen = to-from;
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(dlen*sizeof(cuFloatComplex));
	for(int i = from; i< to; i++)
	{
		cSamples[i%dlen] = make_cuFloatComplex(vSamples[i],0.0);
	}
	return cSamples;
}
//
//
cuFloatComplex* getComplexEmpty(int _dlen)
{
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	/*for(int i = 0; i< _dlen; i++)
	{
		cSamples[i] = make_cuFloatComplex(0.0,0.0);
	}*/
	return cSamples;
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Error Checker forv CUDA Function Calls %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
#define cuErrorCheck(ans) { cuErrorChecker((ans), __FILE__, __LINE__); }
inline void cuErrorChecker(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Stream Processing using CUDA GPUs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk)
{
	//
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	//hdata.resize(dsize); // This must be done outside with data initialization/generation
	//hout.resize(dsize); // Might be a good idea to initialize outside of this function...
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);	
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	// Trying to Create Page-Locked std::vector - No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);//cudaHostAlloc(...);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	
	for(int i = 0; i<dsize; i++)
	{
		cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
    		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% Implementing a New Stream Processor Technique that Utilizes Batched CUFFT, this must increase Performance %%%
//%%% Reduce the number of streams by Performing Optimized Batched CUFFT, this must be Faster...Target is NetRAD %%
//%%% Data Size = chunk*dsize, where dsize = vector size (<= 100 preferably) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% ftpoint = FFT point, We know for sure that Batched CUFFT is performing well (2048x8192 ~= 0.1xx seconds) %%%%
//%%%%*********************************************************************************************************%%%%
#define N_SIGS  1300//
#define SIG_LEN 2048 //
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	//struct timeval t1, t2;
	//gettimeofday(&t1, 0);
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    		NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    		CUFFT_C2C, N_SIGS);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaStreamCreate(&optisdr_streams[i]);
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	//gettimeofday(&t2, 0);
	//double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	//printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//

void readNetRadSamples2(string strFilename, unsigned int nsamples, vector<float> &vsSamples)
{
	//Read
	ifstream oIFS;
	oIFS.open(strFilename.c_str(),ifstream::in);
	if(!oIFS.is_open())
	{	
		cout << "[SDR_DSL_INFO]$ Error unable to open file \"" << strFilename << "\"" << endl;
		oIFS.close();
		exit(1);
	}
	//
	vsSamples.resize(nsamples);
	int i = 0;
	while(i < nsamples)
	{
		oIFS>>vsSamples[i];
		//vsSamples[i] = atof(tmp[i].c_str());
		//printf("%f, ",vsSamples[i]);
		i++;
	}
	oIFS.close();
}
//
cuFloatComplex* getReferenceSignal(int _dlen)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2("../data/rref3sig.dat",_dlen,rSamples);
	readNetRadSamples2("../data/iref3sig.dat",_dlen,iSamples);
	//
	for(int i = 0; i< _dlen; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
	}
	return refsig;
}
//
// This one accepts the return size and file size.
//
cuFloatComplex* getReferenceSignal(int fsize, int _dlen)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2("../data/rref3sig.dat",fsize,rSamples);
	readNetRadSamples2("../data/iref3sig.dat",fsize,iSamples);
	//
	for(int i = 0; i< fsize; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
		// 
	}
	//
	for(int j = (_dlen-fsize); j< _dlen; j++)
	{
		refsig[j] = make_cuFloatComplex(0.0f,0.0f);
	}
	//
	return refsig;
}
//
cuFloatComplex* resizeVector(cuFloatComplex *inp, int oldlen, int newlen)
{
	cuFloatComplex *outp = (cuFloatComplex*)malloc(newlen*sizeof(cuFloatComplex));
	//
	for(int i = 0; i< newlen; i++)
	{
		outp[i] = inp[i%oldlen];
	}
	//
	//outp[0] = make_cuFloatComplex(2.0f,2.0f); // Just for testing: TODO: Remove when done
	//outp[2048] = make_cuFloatComplex(2.0f,2.0f); // Just for testing: TODO: Remove when done
	//outp[4096] = make_cuFloatComplex(2.0f,2.0f); // Just for testing: TODO: Remove when done
	return outp;
}

//
cuFloatComplex* getComplexSamples(vector<short> &vSamples, int from, int to, int chunksize, int outputsize)
{
	int dlen = outputsize*((to-from)/chunksize);
	int skip = outputsize-chunksize;
	//
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(dlen*sizeof(cuFloatComplex));
	//for(int i = 0;i<dlen;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	int idx = 0;
	if(skip>0)
	{
		for(int i = from; i< to; i++)
		{
		
			if(((i%chunksize)==0)&&(i>from))
			{
				idx+=skip; // We skip 
				cSamples[idx] = make_cuFloatComplex((float)vSamples[i],0.0f);
				idx+=1;
			}
			else
			{
				cSamples[idx] = make_cuFloatComplex((float)vSamples[i],0.0f);
				idx+=1;
			}
		}
	}
	return cSamples;
}
//
cuFloatComplex* getComplexEmpty(int _dlen, int chunksize, int outputsize)
{
	int dlen = (_dlen)*(outputsize/chunksize);
	//
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(dlen*sizeof(cuFloatComplex));
	//for(int i = 0;i<dlen;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	//
	return cSamples;
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void spMultiply(vector<cuFloatComplex*> hdata1, vector<cuFloatComplex*> hdata2,vector<cuFloatComplex*> ddata1, vector<cuFloatComplex*> ddata2, vector<cuFloatComplex*> dout, int spsize,int chunk)
{
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	cudaStream_t optisdr_streams[spsize];
	//
	for(int i = 0;i<spsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	for(int i = 0;i<spsize;i++)
	{
		//
		cudaMemcpyAsync(ddata1[i],hdata1[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		cudaMemcpyAsync(ddata2[i],hdata2[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<spsize; i++)
	{		
		//
        	complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],ddata1[i],ddata2[i],chunk);
		cudaStreamSynchronize(optisdr_streams[i]);
		//cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < spsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
	}
	//
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void hMultiply(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	//
	cuFloatComplex *drefsig,*dftrefsig;
	cudaMalloc((void**)&drefsig,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dftrefsig,chunk*sizeof(cuFloatComplex));
	cudaHostRegister(refsig,chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	//
	vector<cuFloatComplex*> ddata,dout,drout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	cudaMemcpyAsync(drefsig,refsig,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[0]);
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		//
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		complexvector_multiply<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dftrefsig,dout[i],chunk);
		// 
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		free(hdata[i]);
	}
	cudaHostUnregister(refsig);
	cudaFree(drefsig);
}
//
void dMultiply(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> dout, int dsize,int chunk)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	//
	cuFloatComplex *drefsig,*dftrefsig;
	cudaMalloc((void**)&drefsig,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dftrefsig,chunk*sizeof(cuFloatComplex));
	cudaHostRegister(refsig,chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	//
	vector<cuFloatComplex*> ddata;//,dout,drout;
	ddata.resize(dsize);
	//dout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	cudaMemcpyAsync(drefsig,refsig,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[0]);
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		//
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		complexvector_multiply<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dftrefsig,dout[i],chunk);
		// 
		//cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		//cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		//cudaFree(dout[i]);
		free(hdata[i]);
	}
	cudaHostUnregister(refsig);
	cudaFree(drefsig);
}
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void hIFFT(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        //
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> ddata,dout,drout;
	ddata.resize(dsize);
	dout.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaStreamSynchronize(optisdr_streams[i]);
		optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],drout[i],ftpoint,chunk);
		//
		cudaMemcpyAsync(hout[i],drout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		cudaFree(drout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	//
}
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//
void dIFFT(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> dout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        //
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> ddata,drout;
	ddata.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		//
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],drout[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		//cudaStreamSynchronize(optisdr_streams[i]);
		optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(drout[i],dout[i],ftpoint,chunk);
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaFree(ddata[i]);
		cudaFree(drout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	//
}
//
//
void ddIFFT(vector<cuFloatComplex*> ddata, vector<cuFloatComplex*> dout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        //
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> drout;
	//ddata.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],drout[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		//cudaStreamSynchronize(optisdr_streams[i]);
		optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(drout[i],dout[i],ftpoint,chunk);
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaFree(ddata[i]);
		cudaFree(drout[i]);
		cufftDestroy(plans[i]);
	}
	//
}
//	
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void dhFFT(vector<cuFloatComplex*> ddata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> dout;
	//
	dout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		//cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		//cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(dout[i]);
		cudaFree(ddata[i]);
		cufftDestroy(plans[i]);
	}
	//
}
//
void hFFT(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	//
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void ddFFT(vector<cuFloatComplex*> ddata, vector<cuFloatComplex*> dout, int dsize,int chunk, int ftpoint)
{
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaFree(ddata[i]);
		//
		cufftDestroy(plans[i]);
	}
	//
}
//
void dFFT(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> dout, int dsize,int chunk, int ftpoint)
{
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> ddata;
	ddata.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		//
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaFree(ddata[i]);
		//cudaFree(dout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	//
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void dhIFFT(vector<cuFloatComplex*> ddata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        //
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	//
	vector<cuFloatComplex*> dout,drout;
	//ddata.resize(dsize);
	dout.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		//cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		//
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Inverse transform fail.\n");}
		//
		//cudaStreamSynchronize(optisdr_streams[i]);
		optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],drout[i],ftpoint,chunk);
		//
		cudaMemcpyAsync(hout[i],drout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		cudaFree(drout[i]);
		
		cufftDestroy(plans[i]);
	}
	//
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void XCorrSP(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	//printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	cuFloatComplex *drefsig,*dftrefsig;
	cudaMalloc((void**)&drefsig,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dftrefsig,chunk*sizeof(cuFloatComplex));
	//cudaHostRegister(refsig,chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	cudaMemcpy(drefsig,refsig,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//cudaMalloc((void**)&drout,chunk*sizeof(cuFloatComplex));
	//
	vector<cuFloatComplex*> ddata,dout,drout;
	ddata.resize(dsize);
	dout.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaStreamCreate(&optisdr_streams[i]);
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	
	res = cufftExecC2C(plans[0],drefsig,dftrefsig,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dftrefsig,drout[i],chunk);
		//
		res = cufftExecC2C(plans[i],drout[i],ddata[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Inverse transform fail.\n");}
		//
		// TODO: Try using the Block Size =128, i.e optisdrifftscale<<<chunk/128,128,0,optisdr_treams[i]>>>(...);
        	optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dout[i],ftpoint,chunk);
		//
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		cudaFree(drout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	cudaHostUnregister(refsig);
	cudaFree(drefsig);
	//gettimeofday(&t2, 0);
	//double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	//printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void hcomplexvector_multiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int index = 0;
	//
	while(index < N)
	{
		outp[index].x = (_cx[index].x*_cy[index].x) - (_cx[index].y*_cy[index].y);//cuCmulf(_cx[index],_cy[index]);
		outp[index].y = (_cx[index].x*_cy[index].y) + (_cx[index].y*_cy[index].x);
		//outp[index] = cuCmulf(_cx[index],_cy[index]);
		index+=1;
	}
}
//
//
void hcomplexvector_multiply2(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N,int ftpoint)
{
	int index = 0;
	//
	printf("\n");
	while(index < ftpoint)
	{
		//outp[index].x = (_cx[index].x*_cy[index].x) - (_cx[index].y*_cy[index].y);//cuCmulf(_cx[index],_cy[index]);
		//outp[index].y = (_cx[index].x*_cy[index].y) + (_cx[index].y*_cy[index].x);
		//printf("[Multiplying: A[%d] = (%f + %f) with B[%d] = (%f + %f) ], ",index,_cx[index].x,_cx[index].y,index,_cy[index].x,_cy[index].y);
		outp[index] = cuCmulf(_cx[index],_cy[index]);
		index+=1;
	}
	printf("\n");
}
//
void complexvector_multiply_h(vector<cuFloatComplex*> inp1,cuFloatComplex* inp2,vector<cuFloatComplex*> outp,int dsize,int N)
{
	int i = 0;
	//
	while(i < dsize)
	{
		hcomplexvector_multiply(inp1[i],inp2,outp[i],N);
		i+=1;
	}
}
//
void complexvector_multiply_h2(vector<cuFloatComplex*> inp1,cuFloatComplex* inp2,vector<cuFloatComplex*> outp,int dsize,int N,int ftpoint)
{
	int i = 0;
	//
	while(i < dsize)
	{
		hcomplexvector_multiply2(inp1[i],inp2,outp[i],ftpoint,ftpoint);
		i+=1;
	}
}
//
//
//
// Creating a Dummy h vector for computing the Hilbert Transform
//
cuFloatComplex* getHilbertHVector(int _dlen)
{
	//
	cuFloatComplex *h = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	h[0] = make_cuFloatComplex(1.0f,0.0f);
	h[_dlen/2] = make_cuFloatComplex(1.0f,0.0f);
	//
	int i = 1;
	int j = (_dlen/2)+1;
	//
	while(i<(_dlen/2))
	{
		h[i] = make_cuFloatComplex(2.0f,0.0f);
		i = i + 1;
		h[j] = make_cuFloatComplex(0.0f,0.0f);
		j = j + 1;
	}
	return h;
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void HilbertSP(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	//blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	//printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//	
	cuFloatComplex* h = resizeVector(getHilbertHVector(ftpoint),ftpoint,chunk);
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	cuFloatComplex *dh;
	cudaMalloc((void**)&dh,chunk*sizeof(cuFloatComplex));
	cudaHostRegister(h,chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	//cudaMalloc((void**)&drout,chunk*sizeof(cuFloatComplex));
	//
	vector<cuFloatComplex*> ddata,dout,dhout,drout;
	ddata.resize(dsize);
	dout.resize(dsize);
	dhout.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		// TODO: Check Out cudaMallocHost(...), cudaMallocManage(...), ...
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable); // Page-Locked Mem.
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaStreamCreate(&optisdr_streams[i]);
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	cudaMemcpyAsync(dh,h,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[0]);
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dhout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dh,dhout[i],chunk);
		//
		res = cufftExecC2C(plans[i],dhout[i],drout[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaMemcpyAsync(hout[i],drout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		cudaFree(dhout[i]);
		cudaFree(drout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	cudaHostUnregister(h);
	cudaFree(dh);
	//
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//
int main()
{
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//%%%%%%%%% Fill a cudaDeviceProp structure with the properties %%%%%%%%%%%%
	//%%%%%%%%% we need our device to have %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	cudaDeviceProp deviceProp;
	//cudaEvent_t start, stop;
	struct timeval t1, t2,strt,endt;
	//%%%%%%%%%%%%%% otherwise use device with highest Gflops/s %%%%%%%%%%%%%%%%
	int cuda_device = 0;
	//
	cudaGetDevice(&cuda_device); // Get Just the CUDA Device name
	cudaSetDevice(cuda_device);
	cudaGetDeviceProperties(&deviceProp, cuda_device);
	printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_device, deviceProp.name);
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//%%%%%%%%%%%%%%%% My Main code starts here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//%%% TODO: 20*1300*2048 = 53248000 = maximum optimized for..loop size. %%%
	//%%% TODO: 
	int dsize = 5; // Maximum for 1300*2048 sp sizes
	int subchunk = 130000/N_SIGS/dsize;
	int chunk = N_SIGS*SIG_LEN; //1300*2048, lets try 6500*2048
	int ftpoint = 2*SIG_LEN; // TODO: Cater for Zero-Padding
	int dataLen = subchunk*dsize*chunk;
	int rssize = 300;
	//vector<cuFloatComplex*> hdata,hout;
	//hdata.resize(dsize);
	//hout.resize(dsize);
	//
	cuFloatComplex* refsig = resizeVector(getReferenceSignal(rssize,ftpoint),ftpoint,(ftpoint/SIG_LEN)*chunk); // From 300 to chunk,
	//cuFloatComplex* refsig2 = resizeVector(getReferenceSignal(rssize,ftpoint), ftpoint, ftpoint);
	vector< vector<cuFloatComplex*> > hdata2,hout2,hout3; // to cater for 0...2048*130000
	hdata2.resize(subchunk);
	hout2.resize(subchunk);
	hout3.resize(subchunk);
	//
	// Read NetRAD Samples Here
	//
	printf("\n[SDR_DSL_INFO]$ Reading NetRAD Data of Size: %i, subchunk = %i .\n",dataLen,subchunk);
	gettimeofday(&strt, 0);
	//
	vector<short> vsNetRadSamples;
	readNetRadSamples("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin",dataLen,vsNetRadSamples);
	// Format Data Into Matrix
	/*for(int i=0; i<dsize; i++)
	{
		int to=i*chunk+chunk, from=i*chunk;
		hdata[i] = getComplexSamples(vsNetRadSamples,from,to); //0...2047, 2048...4095, etc.
		hout[i] = getComplexEmpty(chunk); // Init. output vector
		//printf("Test::%f",hdata[i][0].x); // Testing
	}*/
	
	for(int j=0; j<subchunk; j++)
	{
		int dfrom = dsize*j;
		int dto = dsize*j+dsize;
		vector<cuFloatComplex*> hdata,hout,houtb;
		hdata.resize(dsize);
		hout.resize(dsize);
		houtb.resize(dsize);
		//
		for(int i=dfrom; i<dto; i++)
		{
			int to=i*chunk+chunk, from=i*chunk;
			//hdata[i%dsize] = getComplexSamples(vsNetRadSamples,from,to); //0...2047, 2048...4095, etc.
			hdata[i%dsize] = getComplexSamples(vsNetRadSamples,from,to,SIG_LEN,ftpoint);
			hout[i%dsize] = getComplexEmpty(chunk,SIG_LEN,ftpoint); // Init. output vector
			houtb[i%dsize] = getComplexEmpty(chunk,SIG_LEN,ftpoint);
			//hout[i%dsize] = getComplexEmpty(chunk); // Init. output vector
			//printf("Test::%f",hdata[i][0].x); // Testing
		}
		hdata2[j] = hdata;
		hout2[j]  = hout;
		hout3[j]  = houtb;
	}
	//
	cuFloatComplex* refsig3 = (cuFloatComplex*)malloc(ftpoint*sizeof(cuFloatComplex));
	for(int j = 0; j< ftpoint; j++)
	{
		refsig3[j] = refsig[j];
	}
	//
	//
	gettimeofday(&endt, 0);	
	double dactime = (1000000.0*(endt.tv_sec-strt.tv_sec) + endt.tv_usec-strt.tv_usec)/1000000.0;
	printf("\n[SDR_DSL_INFO]$ Time Taken by Data Reader = %f s.\n",dactime);	
	//
	//
	gettimeofday(&t1, 0);
	//
	// Exec. Stream Processor
    	//streamprocessor(hdata,hout,dsize,chunk);
	//streamprocessor(hdata,hout,dsize,chunk,ftpoint);
	//for(int i = 0; i<subchunk; i++)
	//{
	//	streamprocessor(hdata2[i],hout2[i],dsize,chunk,ftpoint);
	//}
	for(int i = 0; i<subchunk/2; i++)
	{
		XCorrSP(hdata2[i],refsig,hout2[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
		//complexvector_multiply_h2(hout2[i],refsig3,hout3[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
		// HilbertSP(hdata2[i],hout2[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
	}
	//complexvector_multiply_h2(hout2[0],refsig3,hout3[0],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
		
	cudaSetDevice(cuda_device+1);
	cudaGetDeviceProperties(&deviceProp,cuda_device+1);
	printf("[SDR_DSL_INFO]$ Setting Device ID :[%d] Name:[%s]\n",cuda_device+1, deviceProp.name);
	//
	for(int i = subchunk/2; i<subchunk; i++)
	{
		//dFFT(hdata2[i],dout2[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
		XCorrSP(hdata2[i],refsig,hout2[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
		//complexvector_multiply_h2(hout2[i],refsig3,hout3[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
		//HilbertSP(hdata2[i],hout2[i],dsize,(ftpoint/SIG_LEN)*chunk,ftpoint);
	}
	//
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	printf("\n[SDR_DSL_INFO]$ Time Taken by FFT = %f s ~= %f...!\n", time,time2);
	//
     	float *fftOutput = (float*)malloc(chunk*sizeof(float));
      	for (int i = 0; i < 1; i++)//subchunk
      	{
		cuFloatComplex* tmpout = hout2[0][0];//hout2[0][0];//hout2[0][i];//TODO: Uncomment this after testing 
		//cuFloatComplex *tmpout = hdata2[0][i];
		for(int j = 0; j< ftpoint; j++)
		{
			fftOutput[j] = cuCrealf(tmpout[j]);
		}
		printf("[SDR_DSL_INFO]$ Output = [");
		for( int i = 0; i < ftpoint;i++)
		{
			printf("%f , ",fftOutput[i]);
		}
		printf("]\n");
      	}
	//
	// Clean-Up Boy
	for(int j=0;j<subchunk;j++)
	{
		for( int i = 0; i < dsize;i++)
		{
			free(hout2[j][i]); // Clear Host Memory
			//cudaFree(dout2[j][i]);
		}
	}
	//cudaThreadExit();
	//
	free(refsig);
	free(fftOutput);
	cudaDeviceReset();
	//
	return 0;
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Example Test Signals %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void getShortSamples(vector<short> &vSamples, unsigned int nSamples)
{
	//Read
	vSamples.resize(nSamples);
	for(unsigned int i = 0; i< nSamples; i++)
	{
		vSamples[i] = i;
		//printf("%i,",i);
	}
}
//
void getRealSamples(vector<float> &vSamples, unsigned int nSamples)
{
	//Read
	vSamples.resize(nSamples);
	for(unsigned int i = 0; i< nSamples; i++)
	{
		vSamples[i] = (float)(i);
		//printf("%i,",i);
	}
}
//
float* getRealFSamples(int dalen)
{
	//Read
	float *dSamples = (float *)malloc(dalen*sizeof(float));
	for(int i = 0; i< dalen; i++)
	{
		dSamples[i] = (float)(i);
		//printf("%f,",dSamples[i]);
	}
	return dSamples;
}
//
void getRandSamples(vector<float> &vSamples, unsigned int nSamples, int nbound)
{
	//Read
	vSamples.resize(nSamples);
	for(unsigned int i = 0; i< nSamples; i++)
		vSamples[i] = rand()%nbound;
}
//
cuFloatComplex* getComplexSamples(int _dlen)
{
	cuFloatComplex *dSamples = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	for(int i = 0; i< _dlen; i++)
	{
		dSamples[i] = make_cuFloatComplex((float)(i),(float)(i));
	}
	return dSamples;
}
cuFloatComplex* getComplexRandomSamples(int _dlen)
{
	cuFloatComplex *dSamples = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	for(int i = 0; i< _dlen; i++)
	{
		dSamples[i] = make_cuFloatComplex(rand(),rand());
	}
	return dSamples;
}
//
void getAscendingRandSamples(vector<float> &vSamples, unsigned int nSamples, int nbound)
{
	//Read
	vSamples.resize(nSamples);
	srand(time(0));
	for(unsigned int i = 0; i< nSamples; i++)
		vSamples[i] = rand()%nbound;
}
//%%%%%%%%%%%%%%%%%%%%%%% End of FIR Filter Implementation %%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

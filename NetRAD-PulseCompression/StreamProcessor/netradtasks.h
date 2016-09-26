//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: Profiling Task Graph Data Structure %%%%%%%%%%%%%%%%%%%
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
#include <fftw3.h>
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Include Cuda run-time and inline Libraries %%%%%%%%%%%%%%%%%
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>


using namespace std;
//
#define REAL 0
#define IMAG 1
//
enum TASK {IO=0,HT,FFT,MULT,IFFT,SUM};
//
//
enum CE {MCPU=0,GPU,FPGA,PHI};
//

#ifndef __NETRADTASKS__ // Define Class Guards
#define __NETRADTASKS__

class NetRADTasks
{

public:
	//
	//__host__ __device__ void RealComplexProduct(cufftComplex *idata,float *hdata,cufftComplex *odata,int n);
	void performFFTW(float *ftindata,float *ftoutdata,int ftpoint);	
	void performIFFTW(fftw_complex *ftindata,fftw_complex *ftoutdata,int ftpoint);
	void parFFT(vector<fftw_complex*> ftidata, vector<fftw_complex*> ftodata, int ftpoint);
private:
	void copyFFTResults(fftw_complex *result,fftw_complex* ftoutdata);
	void r2CFFTW(fftw_complex *ftindata,float *signal);
};
#endif // End Class Guardsq
//
//
void NetRADTasks::parFFT(vector<fftw_complex*> ftidata, vector<fftw_complex*> ftodata, int ftpoint) // Care full implement this using MPI
{

}
//
void NetRADTasks::performRealFFTW(float *ftindata,fftw_complex ftoutdata,int ftpoint)
{
	int NUM_POINTS = ftpoint;
	fftw_complex signal[NUM_POINTS]; // Declare Signal/Data Holder
	fftw_complex result[NUM_POINTS]; // Declare FFT Results Container

	fftw_plan plan = fftw_plan_dft_1d(NUM_POINTS, signal, result, FFTW_FORWARD, FFTW_ESTIMATE); // Create FFT Plan

	r2CFFTW(ftindata,signal); // Function to Generate/Aqcuire Signal 
	fftw_execute(plan);     // Execute the FFTW with given plan
	copyFFTResults(result,ftoutdata); // Display FFTW Results

	fftw_destroy_plan(plan);
}
//
void NetRADTasks::performIFFTW(fftw_complex *ftindata,fftw_complex *ftoutdata,int ftpoint)
{
	fftw_plan plan = fftw_plan_dft_1d(ftpoint,ftindata,ftoutdata, FFTW_INVERSE, FFTW_ESTIMATE); // Create FFT Plan

	fftw_execute(plan);     //
	//
	fftw_destroy_plan(plan);
}
//
void NetRADTasks::r2CFFTW(float *ftindata,fftw_complex* signal)
{
    int i;
    for (i = 0; i < NUM_POINTS; ++i)
    {
        signal[i][REAL] = ftindata[i];
        signal[i][IMAG] = 0.0;
    }
}
//
void NetRADTasks::copyFFTResults(fftw_complex* result,fftw_complex* ftoutdata)
{
    int i;
    
    for (i = 0; i < NUM_POINTS; ++i)
    {
	ftoutdata[i][REAL] = result[i][REAL];
        ftoutdata[i][IMAG] = result[i][IMAG];
    }
}
//
/*__host__ __device__ void NetRADTasks::RealComplexProduct(cufftComplex *idata,float *hdata,cufftComplex *odata,int n)
{
	//
	int t = blockIdx.x;// + threadIdx.x;
	//int loc_t = threadIdx.x;

	if (t < n)
	{
		odata[t].x = idata[t].x*hdata[t];
		odata[t].y = idata[t].y*hdata[t];
	}
	__syncthreads();
}*/

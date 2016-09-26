//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: OptiSDR CUDA Kernels Simplified for  %%%%%%%%%%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%% Inlcude some C Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef _OPTISDR_CUDA_H_
#define _OPTISDR_CUDA_H_
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include <sys/sysinfo.h>
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Include Cuda run-time and inline Libraries %%%%%%%%%%%%%%%%%
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <omp.h>

using namespace std;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%% Definitions of Some Variables and Constants %%%%%%%%%%%%%%
//Simple square
#define SQR(x) ((x)*(x))
// A macro to do safe division ( checks for a 0 denominator)
#define fDivide(numerator, denominator) (float)denominator==0.0f?0.0f:numerator/(float)denominator
// A macro to do safe division ( checks for a 0 denominator)
#define dDivide(numerator, denominator) (double)denominator==0.0f?0.0f:numerator/(double)denominator
//#define N_SIGS  1300//
#define SIG_LEN 2048 //
#define mPI 3.14159265359f
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% File I/O Functions Definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
extern void readNetRADTextFile(string fln,vector<short> &vsSamples, int N); 
extern int readToBinary(string fln1, string fln2);
extern void readFileToVector(string strFilename, unsigned int numSamples, vector<short> &vsSamples); // This Func
extern void readFileToFloat(string strFilename,unsigned int numSamples,float *vecOut);
extern void readFileToComplex(string strFilename,unsigned int numSamples,cuFloatComplex *outVec,int type);
extern void readNetRadSamples(string strFilename, unsigned int uiNSamples, vector<short> &vsSamples);
extern void getShortSamples(vector<short> &vSamples, unsigned int nSamples);
extern void getRealSamples(vector<float> &vSamples, unsigned int nSamples);
extern float* getRealFSamples(int dalen);
extern void getRandSamples(vector<float> &vSamples, unsigned int nSamples);
extern void getAscendingRandSamples(vector<float> &vSamples, unsigned int nSamples);
extern cuFloatComplex* getComplexSamples(int _dlen);
extern cuFloatComplex* getComplexRandomSamples(int _dlen);
extern cuFloatComplex* getComplexSamples(vector<short> &vSamples, int _dlen);
extern cuFloatComplex* getComplexSamples(vector<short> &vSamples, int from, int to);
extern cuFloatComplex* getComplexEmpty(int _dlen);
extern void readNetRadSamples2(string strFilename, unsigned int nsamples, vector<float> &vsSamples);
extern cuFloatComplex* getReferenceSignal(int _dlen);
extern cuFloatComplex* getReferenceSignal(int fsize, int _dlen);
extern cuFloatComplex* resizeVector(cuFloatComplex *inp, int oldlen, int newlen);
extern cuFloatComplex* getComplexSamples(vector<short> &vSamples, int from, int to, int chunksize, int outputsize);
extern cuFloatComplex* getComplexEmpty(int _dlen, int chunksize, int outputsize);
//
extern cuFloatComplex* getHilbertHVector(int _dlen);
extern cuFloatComplex* getChunk(cuFloatComplex *inp, int chunk, int from, int to);
extern cuFloatComplex* getZeroPadded(cuFloatComplex *inp, int initlen, int newlen, int skip);
extern void append(cuFloatComplex *A, cuFloatComplex *B, int sizeA, int sizeB, int from);
//
extern cuFloatComplex* getChunk(cuFloatComplex *inp, int from, int to);
extern void writeFileF(const char *fpath, float *data,	const unsigned int len);
extern void writeFileF(const char *fpath,cuFloatComplex *xdata, const unsigned int len);
extern cuFloatComplex* getChunk(vector<short> &inp, int from, int to);
extern void vresize(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip);
extern cuFloatComplex* getReferenceSignal(string fnamer, string fnamei, int _dlen);
extern cuFloatComplex* getReferenceSignal(string fnamer, string fnamei, int _dlen,int ftpoint);
extern void repMat(cuFloatComplex *dhout,cuFloatComplex *drout,int hmsize,int chunks,int padding);
extern void zepadMat(cuFloatComplex *dhout,cuFloatComplex *drout,int row,int col,int padding);
extern cuFloatComplex* hamming(int N);
extern cuFloatComplex* interFilterCoef(int M, int fIdx);
extern cuFloatComplex* hamming(cuFloatComplex *inp,int N,int ftpoint);
extern void _meanSqr(cuFloatComplex* hdata, int M);
extern void TestStuff();
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%% Simplified Signal Processing Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
extern __global__ void _10logabs(cuFloatComplex *invec, cuFloatComplex *out, int N);
extern __global__ void complexvector_multiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_multiply1d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_multiply2d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_multiply3d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_conjugate(cuFloatComplex *_cx, cuFloatComplex *outp,int N);
extern __global__ void complexvector_abs(cuFloatComplex *_cx,float *outp,int N);
extern __global__ void optisdrifftscale(cuFloatComplex *invec, cuFloatComplex *out, int fp, int N);
extern __global__ void xcorrmultiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void _hamming(cuFloatComplex *outp, int N);
extern __global__ void repmat(cuFloatComplex *dhout,cuFloatComplex *drout,int hmsize,int chunks);
extern __global__ void ones(cuFloatComplex *outd,int N, int offset);
extern __global__ void zeros(cuFloatComplex *outd,int N, int offset);
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Simplified DSP CUDA Kernel Calls %%%%%%%%%%%%%%%%%%%%%%
//
extern double _10logAbs(vector<cuFloatComplex*> hdata, int dsize,int chunk,cuFloatComplex Vpp,cuFloatComplex div, float logVal);
extern double BatchedFFT(vector<cuFloatComplex*> hdata,int dsize,int chunk, int ftpoint);
extern double _10logAbs(vector<cuFloatComplex*> hdata, int dsize,int chunk);
extern double HilbertSP(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
extern double XCorrSP(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
extern double XCorrSP2(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
extern void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
extern double _JSSHamming(vector<cuFloatComplex*> hdata, int dsize,int chunk,int hmsize, int padding);
extern double _interferenceFilter(cuFloatComplex* hdata, float fc, float fs, int M);
extern double _fft(cuFloatComplex* hdata, int M);
extern cudaError_t cuda_main();
//
//
#endif

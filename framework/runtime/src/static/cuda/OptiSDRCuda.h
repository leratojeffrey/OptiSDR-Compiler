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
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Include Cuda run-time and inline Libraries %%%%%%%%%%%%%%%%%
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <omp.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/sort.h>
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%% Include Delite Tools %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include "cudahelperFuncs.h"
#include "cudaDeliteArrayfloat.h"
#include "cudaDeliteArraydouble.h"
#include "cudaDeliteArrayint32_t.h"
#include "cppDeliteArrayfloat.h"
//#include "cudaDeliteArray.h"
#include "cppDeliteArraydouble.h"
//#include "qcustomplot.h"
//
using namespace std;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%% OptiSDR USRP1 Functions Definitions %%%%%%%%%%%%%%%%%%%%%%
// TODO: This is where comments started /*
//
#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
#include <csignal>
#include <complex>
//
//using namespace boost::program_options;
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% OptiSDR USRP Module Functions Declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

struct optisdr_usrp_config
{
    uhd::usrp::multi_usrp::sptr usrp;
    std::string format;
    std::string wirefmt;//
    std::string file; ////
    std::string type;
    size_t spb;
    size_t total_num_samps;
    double total_time;
    bool bw_summary;
    bool stats;
    bool null;
    bool enable_size_map;
    bool continue_on_bad_packet;
    cuFloatComplex *outdata;//
};
//
extern struct optisdr_usrp_config configs;
extern void readData(cuFloatComplex *outdata, std::vector<std::complex<short> > &buffer, unsigned int from, unsigned int to);
extern void recv_to_file(uhd::usrp::multi_usrp::sptr usrp, std::string &cpu_format, std::string &wire_format,cuFloatComplex *outdata, size_t samps_per_buff, unsigned long long num_requested_samples, double time_requested, bool bw_summary, bool stats,
    bool null, bool enable_size_map, bool continue_on_bad_packet);
extern double streamin_usrp();
extern double initUSRP(int argc, char *argv[]);
//
//
extern void recv_whileloop(uhd::usrp::multi_usrp::sptr usrp, std::string &cpu_format, std::string &wire_format,cuFloatComplex *outdata, size_t samps_per_buff, unsigned long long num_requested_samples, double time_requested, bool bw_summary, bool stats,
    bool null, bool enable_size_map, bool continue_on_bad_packet);
extern void rx_while();
extern void usrpstreamer(int flag,cudaDeliteArrayfloat x,cudaDeliteArrayfloat y);
extern void usrpstreamer(int flag,cudaDeliteArrayfloat x,cudaDeliteArrayfloat y, double freq);
extern void usrpstream(int flag,cudaDeliteArrayfloat x,cudaDeliteArrayfloat y);
extern void initusrp(int arg);
extern void startusrp(int tst, cppDeliteArraydouble in, cppDeliteArraydouble out);
extern double startUSRP(int argc, char *argv[], double Fc, double Fs, double Gn,double Bw, int Ns);
//
//TODO: Comments end here */
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%% Definitions of Some Variables and Constants %%%%%%%%%%%%%%
//#define N_SIGS  1300
//#define SIG_LEN 2048
//
// TODO: Generalize these stuff here for the sp, rename to make sense
//

struct StreamProcessorConfig
{
	int dsize,
			chunk,
			subchunk,
			ftpoint,
			dataLen,
			NUM_SIGS;
	vector< vector<cuFloatComplex*> > hout1; //Memory Vectors
	vector<cuFloatComplex*> hdata0,hdata1,hdata2; // Memory Vectors
	cuFloatComplex* refsig;
};
//
extern StreamProcessorConfig Configs;
extern void initStreamProcessor(int len, int _dsize,int numsigs, int ftp);
extern void zpad(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip);
extern void resize(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip);
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% File I/O Functions Definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%% Simplified Signal Processing Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
extern __global__ void complex2delitearray(cuFloatComplex *invec,cudaDeliteArrayfloat outrl,cudaDeliteArrayfloat outim,int N);
extern __global__ void complexvector_multiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_multiply1d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_multiply2d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_multiply3d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void complexvector_conjugate(cuFloatComplex *_cx, cuFloatComplex *outp,int N);
extern __global__ void complexvector_abs(cuFloatComplex *_cx,float *outp,int N);
extern __global__ void optisdrifftscale(cuFloatComplex *invec, cuFloatComplex *out, int fp, int N);
extern __global__ void optisdrifftscale(cuFloatComplex *invec,cudaDeliteArrayfloat outrl,cudaDeliteArrayfloat outim,int fp, int N);
extern __global__ void xcorrmultiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N);
extern __global__ void zrpad(cudaDeliteArrayfloat inp,cudaDeliteArrayfloat outp,int ncol, int ocol);
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Simplified DSP CUDA Kernel Calls %%%%%%%%%%%%%%%%%%%%%%
//
extern void HilbertSP(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
extern void XCorrSP(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
extern void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint);
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%% Simplified Signal Processing Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
extern void fftq(double *x,double *out, double *y, int SIG_LEN);
extern void fftp(cppDeliteArrayfloat x,cppDeliteArrayfloat y,cppDeliteArrayfloat out, int SIG_LEN);
//
extern void hilbert(cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim, int ftpoint);
extern void hilbert2(int ftpoint, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim);
extern void fftc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cudaDeliteArrayfloat out, int SIG_LEN);
extern void fftx(cudaDeliteArrayfloat x, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat outim, int SIG_LEN);
extern void ifftx(cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim, int SIG_LEN);
extern void fftx2( int SIG_LEN, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim);
extern void ifftx2(int SIG_LEN, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim);
extern void ifftc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cudaDeliteArrayfloat out, int SIG_LEN);
extern void xcorr(cudaDeliteArrayfloat x1, cudaDeliteArrayfloat x2, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y1, cudaDeliteArrayfloat y2, cudaDeliteArrayfloat outim, int ftpoint);
extern void xcorr(int ftpoint, cudaDeliteArrayfloat x1, cudaDeliteArrayfloat x2, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y1, cudaDeliteArrayfloat y2, cudaDeliteArrayfloat outim);
extern void xcorr(int ftpoint, cudaDeliteArrayfloat x1, cudaDeliteArrayfloat x2, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y1, cudaDeliteArrayfloat y2, cudaDeliteArrayfloat outim,int ftp);

//
extern void Absv(int M, cudaDeliteArrayfloat x, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat outim);
extern void Sqrtv(cudaDenseVectorFloat x,cudaDenseVectorFloat out, int M);
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% OptiSDR Stream Processor Block Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//enum StreamProcessorOps {Hilbert=0,Fft,Ifft,Xcorr,Conv,Ddc,Log10,Abs,Psd,Streamprocessor};
enum StreamProcessorOps /*OptiSDR_DSP*/ {Hilbert= 0,Fft,XCorr,Ifft,Mult,Psd,Log10,Sin,Cos,Abs,Hamming,Convencode,Qam,Qpsk,
	  Cycprefix,Fread,Freadc,Fwrite,Modulate,Dct,Ofdm,Streamprocessor,Ddc,Conv};
//%
extern void exec_c2c_dsp(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat outim, int funcID, int flen);
//extern void exec_c2c_dsp_mod(cudaDeliteArrayfloat x1,cudaDeliteArrayfloat y1,cudaDeliteArrayfloat x2,cudaDeliteArrayfloat y2,cudaDeliteArrayfloat out, int funcID);
extern void streamprocessor( cudaDeliteArrayfloat x,cudaDeliteArrayfloat y, 
			   cudaDeliteArrayfloat funcIDs, cudaDeliteArrayfloat funcPARs,
			   cudaDeliteArrayfloat outx, cudaDeliteArrayfloat outy);
extern void parallelstreamer(int xC,cudaDeliteArrayfloat* src,cudaDeliteArrayint32_t* funcs,cudaDeliteArrayfloat* outx,cudaDeliteArraydouble* pars,cudaDeliteArrayfloat* outy, int size);
//
//
//extern void fftc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat out, int SIG_LEN);
//extern void ifftc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat out, int SIG_LEN);
#endif

#ifndef _OPTISDR_CUDA_
#define _OPTISDR_CUDA_
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include "OptiSDRCuda.h"
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%% OptiSDR Suppot Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%
//%%%%%% Read Text File to Binary file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int readToBinary(string fln1, string fln2)
{
	std::ifstream in(fln1.c_str()); // Create a simple Text File Reader - C++
	std::ofstream out(fln2.c_str(),std::ios::binary); // Create a simple C++ Binary File Writer

	int d; // We read as Integers, well we can make short if we want
	while(in >> d) // Untill end of file
	{
		out.write((char*)&d, sizeof d); // Write into output binary file
		//cout<<d<<endl; // Uncomment to display values
	}
	// Good, now we have a .bin file that GNURadio can read.
	return 0;
}
//%
void readNetRADTextFile(string fln,vector<short> &vsSamples, int N)
{
	vsSamples.resize(N);
	std::ifstream in(fln.c_str()); // Create a simple Text File Reader - C++
	//
	int i = 0;
	short d; // We read as Short
	while(in >> d) // Visit every element untill end of file
	{
		vsSamples[i] = d;
		i++;
		//out.write((char*)&d, sizeof d); // Write into output binary file
		//cout<<d<<endl; // Uncomment to display values
	}
	cout<<endl<<i<<endl;
	// Good, done..
}
//%
//%%%
//%
//double GPUcalTime = 0.0;
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
	//
	vsSamples.resize(uiNSamples);
	//
	oIFS.read((char*)&vsSamples.front(), sizeof(short) * uiNSamples);

	if(oIFS.gcount() << sizeof(short) * uiNSamples && oIFS.eof())
	{
		cout << "[SDR_DSL_INFO]$ Warning: hit end of file after " << oIFS.gcount() / sizeof(short) << " samples. Output is shortened accordingly." << endl;
		vsSamples.resize(oIFS.gcount() / sizeof(short));
	}
	//
	oIFS.close();
}
//
void readFileToVector(string strFilename, unsigned int numSamples, vector<short> &vsSamples)
{
	//Read File into Complex Data
	//
	vsSamples.resize(numSamples);	
	int dataSize=numSamples*sizeof(short);
	short *buffer = (short*)malloc(dataSize);
	//
	struct timeval t3,t4;
	gettimeofday(&t3, 0);
	FILE *fileIO;
	fileIO=fopen(strFilename.c_str(),"r");
	if(!fileIO)
	{
		printf("[SDR_DSL_ERROR]$ Unable to open file!");
		exit(1);
	}	
	fread(buffer,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
	fclose(fileIO);
	//
	gettimeofday(&t4, 0);
	double time1 = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000000.0;
	printf("[SDR_DSL_INFO]$ Overall Reading Data Into Memory = %f s .\n", time1);
	//
	#pragma omp parallel for
	for(int i=0; i<numSamples; i++)
	{
		vsSamples[i] = buffer[i];
	}
	//
	free(buffer);
	//
}
//
void readFileToComplex(string strFilename, unsigned int numSamples, cuFloatComplex *outVec, int type)
{
	//Read File into Complex Data
	outVec = (cuFloatComplex*)malloc(numSamples*sizeof(cuFloatComplex));
	if(type == 1)
	{
		int dataSize=numSamples*sizeof(short);
		short *buffer = (short*)malloc(dataSize);
		FILE *fileIO;
		fileIO=fopen(strFilename.c_str(),"r");
		if(!fileIO)
		{
			printf("[SDR_DSL_ERROR]$ Unable to open file!\n");
			exit(1);
		}	
		fread(buffer,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
		fclose(fileIO);
		//
		#pragma omp parallel for
		for(int i=0; i<numSamples; i++)
		{
			outVec[i] = make_cuFloatComplex((float)buffer[i],0.0f);
		}
		//
		free(buffer);
	}
	else if(type == 2)
	{
		int dataSize=numSamples*sizeof(float);
		float *buffer = (float*)malloc(dataSize);
		FILE *fileIO;
		fileIO=fopen(strFilename.c_str(),"r");
		if(!fileIO)
		{
			printf("[SDR_DSL_ERROR]$ Unable to open file!");
			exit(1);
		}	
		fread(buffer,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
		fclose(fileIO);
		//
		#pragma omp parallel for
		for(int i=0; i<numSamples; i++)
		{
			outVec[i] = make_cuFloatComplex(buffer[i],0.0f);
		}	
		//
		free(buffer);
	}
	else if(type == 3)
	{
		int dataSize=numSamples*sizeof(double);
		double *buffer = (double*)malloc(dataSize);
		FILE *fileIO;
		fileIO=fopen(strFilename.c_str(),"r");
		if(!fileIO)
		{
			printf("[SDR_DSL_ERROR]$ Unable to open file!");
			exit(1);
		}	
		fread(buffer,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
		fclose(fileIO);
		//
		#pragma omp parallel for
		for(int i=0; i<numSamples; i++)
		{
			outVec[i] = make_cuFloatComplex((float)buffer[i],0.0f);
		}	
		//
		free(buffer);
	}
	else
	{
		int dataSize=numSamples*sizeof(int);
		int *buffer = (int*)malloc(dataSize);
		FILE *fileIO;
		fileIO=fopen(strFilename.c_str(),"r");
		if(!fileIO)
		{
			printf("[SDR_DSL_ERROR]$ Unable to open file!");
			exit(1);
		}	
		fread(buffer,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
		fclose(fileIO);
		//
		#pragma omp parallel for
		for(int i=0; i<numSamples; i++)
		{
			outVec[i] = make_cuFloatComplex((float)buffer[i],0.0f);
		}	
		//
		free(buffer);
	}
}
//
//
void readFileToFloat(string strFilename, unsigned int numSamples, float *outVec)
{
	//Read File into Complex Data
	outVec = (float*)malloc(numSamples*sizeof(float));
	int dataSize=numSamples*sizeof(float);
	//
	FILE *fileIO;
	fileIO=fopen(strFilename.c_str(),"r");
	if(!fileIO)
	{
		printf("[SDR_DSL_ERROR]$ Unable to open file!");
		exit(1);
	}	
	fread(outVec,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
	fclose(fileIO);
	//
}
//
//
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
	for(int i = 0; i< _dlen; i++)
	{
		cSamples[i] = make_cuFloatComplex(0.0,0.0);
	}
	return cSamples;
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
	readNetRadSamples2("data/rref3sig2.dat",_dlen,rSamples);
	readNetRadSamples2("data/iref3sig2.dat",_dlen,iSamples);
	//
	for(int i = 0; i< _dlen; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
	}
	return refsig;
}
//
//
cuFloatComplex* getReferenceSignal(string fnamer, string fnamei, int _dlen)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2(fnamer,_dlen,rSamples);
	readNetRadSamples2(fnamei,_dlen,iSamples);
	//
	for(int i = 0; i< _dlen; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
	}
	//printf("\n\n%f\n\n\n",refsig[_dlen-1]);
	return refsig;
}
//
//
//
cuFloatComplex* getReferenceSignal(string fnamer, string fnamei, int _dlen, int ftpoint)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(ftpoint*sizeof(cuFloatComplex));
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2(fnamer,_dlen,rSamples);
	readNetRadSamples2(fnamei,_dlen,iSamples);
	//
	for(int i = 0; i< _dlen; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
	}
	for(int i = _dlen; i< ftpoint; i++)
	{
		refsig[i] = make_cuFloatComplex(0.0f,0.0f);
	}
	//printf("\n\n%f\n\n\n",refsig[_dlen-1]);
	return refsig;
}
//
//
// This one accepts the return size and file size.
//
cuFloatComplex* getReferenceSignal(int fsize, int _dlen)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2("data/rref3sig2.dat",fsize,rSamples);
	readNetRadSamples2("data/iref3sig2.dat",fsize,iSamples);
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
	for(int i = 0;i<dlen;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	//
	return cSamples;
}
//
cuFloatComplex* getChunk(vector<short> &inp, int from, int to) // Want to get complex values from shorts vector
{
	//
	int chunk = to - from;
	cuFloatComplex *outp = (cuFloatComplex*)malloc(chunk*sizeof(cuFloatComplex));
	//	
	int index = 0;
	for(int i = from; i< to; i++)
	{
		outp[index] = make_cuFloatComplex((float)inp[i],0.0f);
		index+=1;
	}
	//
	//
	return outp;
}
//
//
cuFloatComplex* getChunk(cuFloatComplex *inp, int from, int to)
{
	//
	int chunk = to-from;
	cuFloatComplex *outp = (cuFloatComplex*)malloc(chunk*sizeof(cuFloatComplex));
	//
	copy(inp + from,inp + to, outp + 0);
	//
	return outp;
}

//
cuFloatComplex* getChunk(cuFloatComplex *inp, int chunk, int from, int to)
{
	//
	//
	cuFloatComplex *outp = (cuFloatComplex*)malloc(chunk*sizeof(cuFloatComplex));
	//
	//copy(inp + from,inp + to, outp + 0);
	int index = 0;
	for(int i = from; i< to; i++)
	{
		outp[index] = inp[i];
		index+=1;
	}
	//
	return outp;
}
//
cuFloatComplex* getZeroPadded(cuFloatComplex *inp, int initlen, int newlen, int skip)
{
	//
	//
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(newlen*sizeof(cuFloatComplex));
	//
	int idx = 0;
	if(skip>0)
	{
		for(int i = 0; i< initlen; i++)
		{
		
			if(((i%skip)==0)&&(i>0))
			{
				idx+=skip; // We skip 
				cSamples[idx] = inp[i];
				idx+=1;
			}
			else
			{
				cSamples[idx] = inp[i];
				idx+=1;
			}
		}
	}
	return cSamples;
}
//
void append(cuFloatComplex *A, cuFloatComplex *B, int sizeA, int sizeB, int from)
{
	int index = 0;
	if((sizeA-sizeB)>=sizeB)
	{
		for(int i=from;i<sizeB;i++)
		{
			A[i] = B[index];
			index++;
		}		
		//printf("index = %d\n",index);
	}
}
//
void writeFileF(const char *fpath, float *data,	const unsigned int len)
{
    printf("[SDR_DSL_INFO]$ Output file: %s\n", fpath);
    FILE *fo;

    unsigned int i=0;

    if ( (fo = fopen(fpath, "w")) == NULL) {printf("[SDR_DSL_INFO]$ IO Error\n"); /*return(CUTFalse);*/}

    for (i=0; i<len; ++i)
    {
	if ( (fprintf(fo,"%.7e\n", data[i])) <= 0 )
	{
	    printf("[SDR_DSL_INFO]$ File write Error.\n");
	    fclose(fo);
	    //return(CUTFalse);
	}
    }

    fclose(fo);
    //return(CUTTrue);
}
//
void writeFileF(const char *fpath,cuFloatComplex *xdata, const unsigned int len)
{
    printf("[SDR_DSL_INFO]$ Output file: %s\n", fpath);
    FILE *fo;

    unsigned int i=0;

    if ( (fo = fopen(fpath, "w")) == NULL) {printf("[SDR_DSL_INFO]$ IO Error\n");}

    for (i=0; i<len; ++i)
    {
	//if((fprintf(fo,"%.7e + %.7ei\n",cuCrealf(xdata[i]),cuCimagf(xdata[i]))) <= 0 )
	if((fprintf(fo,"%f \n",cuCabsf(xdata[i]))) <= 0 )
	{
		printf("[SDR_DSL_INFO]$ File write Error.\n");
		fclose(fo);
	}
    }

    fclose(fo);
    //return(CUTTrue);
}
//
void vresize(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip)
{
	int index = 2*skip;
	int iters = inlen/skip;
	//printf("%d",iters);
	for(int i=0; i<iters; i++)
	{
		for(int j=0; j<skip; j++)
		{
			outp[i*index + j] = inp[j];
		}
	}
	//
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
		outp[index].y = (_cx[index].x*_cy[index].y) + (_cx[index].y*_cy[index].x); //
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
__global__ void _10logabs(cuFloatComplex *_cx, cuFloatComplex *outp, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<N)
		outp[tid].x = 10.0f*logf(cuCabsf(_cx[tid]));
}
//
// float Vpp,float div, float logVal
//
__global__ void _20log10fabs(cuFloatComplex *_cx, cuFloatComplex *outp, int N,cuFloatComplex Vpp,cuFloatComplex divV,float logV)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<N)
	{
		outp[tid].x = logV*log10f(cuCabsf(cuCdivf(cuCmulf(_cx[tid],Vpp),divV))); // logV*log10f(cuCabsf(_cx[tid]));
		outp[tid].y = 0.0f;
	}
}
//
//
__global__ void optisdrifftscale(cuFloatComplex *invec, cuFloatComplex *out, int fp, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		out[tid] = make_cuFloatComplex(cuCrealf(invec[tid])/(float)fp,cuCimagf(invec[tid])/(float)fp);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Simplified DSP CUDA Kernel Calls %%%%%%%%%%%%%%%%%%%%%%
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Interference Filter Using CUDA Stream Processing %%%%%%%%%%%%%
//
// The Interference Filter Kernel
__global__ void ones(cuFloatComplex *outd,int N, int offset)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		outd[tid+offset] = make_cuFloatComplex(1.0f,1.0f);
}
//
__global__ void zeros(cuFloatComplex *outd,int N, int offset)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		outd[tid+offset] = make_cuFloatComplex(0.0f,0.0f);
}
//
// Interference Filter Stream Processor
//
//
cuFloatComplex* interFilterCoef(int M, int fIdx)
{
	cuFloatComplex *cSamples = (cuFloatComplex*)malloc(M*sizeof(cuFloatComplex));
	for(int i = 0; i< M; i++)
	{
		cSamples[i] = make_cuFloatComplex(1.0f,1.0f);
	}
	//1:fIdx+1
	for(int i=0;i<fIdx+1;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	for(int i=M-fIdx+1;i<M;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	return cSamples;
}
//
void _meanSqr(cuFloatComplex* hdata, int M)
{
	float mean= 0.0;
	
	for(int i=0;i<M;i++)
	{
		mean+=pow(cuCabsf(hdata[i]),2);
	}
	float meansqr = sqrt(mean/M);
	for(int i=0;i<M;i++)
	{
		hdata[i].x = hdata[i].x/meansqr;
		hdata[i].y = hdata[i].y/meansqr;
	}
	//return hdata;
}
//
//
double _interferenceFilter(cuFloatComplex* hdata, float fc, float fs, int M) // 
{
	//
	struct timeval kernt1, kernt2;
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(M/(float)threadsPerBlock); // For 1-D Grid
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
	float Df = fs/(float)M; //the freuqency step
	int fIdx = (int)ceil(fc/Df); //the index of the cut-off freq, it is set to the closest larger integer
	printf(" %d. \n",fIdx);
	//	
	cuFloatComplex* h = interFilterCoef(M,fIdx);//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {M};
	int sigLen = M/M; // Make sure chunk is multiple of 2, better done at DSL level
	//
	cuFloatComplex *dh;
	cudaMalloc((void**)&dh,M*sizeof(cuFloatComplex));
	//cudaHostRegister(h,M*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	//cudaMalloc((void**)&drout,chunk*sizeof(cuFloatComplex));
	//
	cuFloatComplex* ddata,*dout,*dhout,*drout;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,M, //advanced data layout, NULL shuts it off
    		NULL, 1,M, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
	// Trying to Create Page-Locked std::vector
	// 
	//
	cudaMemcpy(dh,h,M*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	cudaMalloc((void**)&ddata,M*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout,M*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dhout,M*sizeof(cuFloatComplex));
	cudaMalloc((void**)&drout,M*sizeof(cuFloatComplex));
	//
	cudaMemcpy(ddata,hdata,M*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	//
	gettimeofday(&kernt1, 0);
	// Execution
	//
	res = cufftExecC2C(plan,ddata,dout,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout,dh,dhout,M);
	//
	res = cufftExecC2C(plan,dhout,drout,CUFFT_INVERSE);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	optisdrifftscale<<<dimGrid,dimBlock>>>(drout,dout,M,M);
	//
	//
	gettimeofday(&kernt2, 0);
	double GPUTime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec); ///1000000.0;
	//
	cudaMemcpy(hdata,dout,M*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
	// Releasing Computing Resources
	cudaFree(ddata);
	cudaFree(dout);
	cudaFree(dhout);
	cudaFree(drout);
	//free(hdata);
	cufftDestroy(plan);
	cudaFree(dh);
	free(h);
	//
	return GPUTime;
}
//

//
double _fft(cuFloatComplex* hdata, int M) // 
{
	//
	struct timeval kernt1, kernt2;
	//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {M};
	int sigLen = M/M; // Make sure chunk is multiple of 2, better done at DSL level
	//
	//
	//
	//
	cuFloatComplex* ddata,*dout;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,M, //advanced data layout, NULL shuts it off
    		NULL, 1,M, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
	//
	// 
	//
	cudaMalloc((void**)&ddata,M*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout,M*sizeof(cuFloatComplex));
	//
	cudaMemcpy(ddata,hdata,M*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	//
	gettimeofday(&kernt1, 0);
	// Execution
	//
	res = cufftExecC2C(plan,ddata,dout,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//
	gettimeofday(&kernt2, 0);
	double GPUTime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec); ///1000000.0;
	//
	cudaMemcpy(hdata,dout,M*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
	// Releasing Computing Resources
	cudaFree(ddata);
	cudaFree(dout);
	//free(hdata);
	cufftDestroy(plan);
	//
	return GPUTime;
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Hamming Window using CUDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// TODO: Use 2D and 3D to cater for bigger sizes
//
__global__ void _hamming(cuFloatComplex *outp, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<N)
		outp[tid].x = 0.54f - 0.46f * cos(2.0f*mPI*(tid)/(float)(N-1));
}
//
__global__ void repmat(cuFloatComplex *dhout,cuFloatComplex *drout,int hmsize,int chunks)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<hmsize)
	{
		//for(int i = 0; i<hmsize; i++)
		//{
			drout[tid] = dhout[tid];
		//}
	}
}
//
void repMat(cuFloatComplex *dhout,cuFloatComplex *drout,int hmsize,int chunks,int padding) // e.g. repMat(in,out,2048,2048*1300,2);
{
	int tid = 0;
	int col = padding*hmsize;
	int row = chunks/hmsize;
	while(tid<row)
	{
		for(int i = 0; i<hmsize; i++)
		{
			drout[tid*col+i] = dhout[i];
		}
		tid++;
	}
}
//
void zepadMat(cuFloatComplex *dhout,cuFloatComplex *drout,int row,int col,int padding) // e.g. zepadMat(in,out,1300,2048,2);
{
	int tid = 0;
	int ncol = padding*col;
	//int row = padding*chunks/hmsize;
	while(tid<row)
	{
		for(int i = 0; i<col; i++)
		{
			drout[tid*ncol+i] = dhout[tid*col+i];
		}
		tid++;
	}
}
//
cuFloatComplex* hamming(int N)
{
	cuFloatComplex *outp = getComplexEmpty(N);
	int tid = 0;
	while(tid<N)
	{
		outp[tid].x = 0.54f - 0.46f * cos(2.0f*mPI*(tid)/(float)(N-1));
		tid++;
	}
	return outp;
}
//
//
cuFloatComplex* hamming(cuFloatComplex *inp,int N,int ftpoint)//
{
	cuFloatComplex *outp = getComplexEmpty(ftpoint);
	int tid = 0;
	while(tid<N)
	{
		outp[tid].x = 0.54f - 0.46f * cos(2.0f*mPI*(tid)/(float)(N-1));
		outp[tid] = cuCmulf(inp[tid],outp[tid]);
		tid++;
	}
	for(int i=N;i<ftpoint;i++) outp[i]=make_cuFloatComplex(0.0f,0.0f);
	return outp;
}
//
//
void TestStuff()
{
	//int cuda_device=0;
	//
	////cudaGetDevice(&cuda_device); // Get Just the CUDA Device name
	//cudaSetDevice(cuda_device);
	struct timeval t1,t2;
	//
	int ftpoint = 2048;
	int chunk = ftpoint*1300;
	//dim3 dimBlock, dimGrid;
    	//int threadsPerBlock, blocksPerGrid;
	//dimBlock.x = 1024;
	//threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
	//int grid1 = ceil(chunk/(float)threadsPerBlock);
	//dimGrid.x = grid1;
	//blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	//
	gettimeofday(&t1, 0);
	cuFloatComplex *khout = getComplexEmpty(chunk);
	//cuFloatComplex /*kddata,*kddout,*/ *krefsig =  getComplexEmpty(2*chunk);
	//repMat(getReferenceSignal("data/rref3sig2.dat","data/iref3sig2.dat",100,ftpoint),khout,ftpoint,chunk,1);
	repMat(hamming(ftpoint),khout,ftpoint,chunk,1);
	//zepadMat(khout,krefsig,1300,ftpoint,2);
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("\n\n %f \n\n",time);
	//
	// TODO: Allocate and Init ddata n ddout
	//cudaMalloc((void**)&kddata,ftpoint*sizeof(cuFloatComplex));
	//cudaMalloc((void**)&kddout,chunk*sizeof(cuFloatComplex));
	//cudaMemcpy(krefsig,kddata,ftpoint*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	//repmat<<<dimGrid,dimBlock>>>(kddata,kddout,ftpoint,chunk);
	//
	//cudaMemcpy(khout,kddata,chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
	//
	//
	printf("\n\n[");
	for(int i=0; i<2*ftpoint;i++)
	{
		printf("%f  ",cuCrealf(khout[i]));
		if((i>0) && (i%ftpoint==0))printf("]\n\n[");
	}
	printf("]\n\n");
	//
	//cudaFree(kddata);
	//cudaFree(kddout);
	free(khout);
	//free(krefsig);
	//cudaDeviceReset();
}
//
//%%%%%%%%% Power Spectrum Stream Processing using CUDA GPUs %%%%%%%%%%%%%
//
double _JSSHamming(vector<cuFloatComplex*> hdata, int dsize,int chunk,int hmsize, int padding)
{
	//
	struct timeval t1, t2;
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        //
	int grid1 = ceil(padding*chunk/(float)threadsPerBlock);
	dimGrid.x = grid1;
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	//hdata.resize(dsize); // This must be done outside with data initialization/generation
	//hout.resize(dsize); // Might be a good idea to initialize outside of this function...
	cudaStream_t optisdr_streams[dsize];	
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//drout.resize(dsize);
	//
	cuFloatComplex *dhin;	
	//
	cuFloatComplex *khout =  getComplexEmpty(padding*chunk);
	repMat(hamming(hmsize),khout,hmsize,chunk,padding); // Create Hamming and repeat up to chunk size...
	//
	cudaMalloc((void**)&dhin,padding*chunk*sizeof(cuFloatComplex));
	cudaMemcpy(dhin,khout,padding*chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	// Trying to Create Page-Locked std::vector - No need for this if we need simple malloc()...
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],padding*chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);//cudaHostAlloc(...);
		//cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaMalloc((void**)&ddata[i],padding*chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],padding*chunk*sizeof(cuFloatComplex));
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Execution
	//
	for(int i = 0; i<dsize; i++)
	{
		cudaMemcpyAsync(ddata[i],hdata[i],padding*chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	// Execution
	//	
	gettimeofday(&t1, 0);
	//
	for(int i = 0; i<dsize; i++)
	{		
		//
		//
		complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dhin,ddata[i],dout[i],padding*chunk);
		//
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&t2, 0);
	double GPUTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec); ///1000000.0;
	//
	for(int i = 0; i<dsize; i++)
	{		
		//
		cudaMemcpyAsync(hdata[i],dout[i],padding*chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		//cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		//
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		
		//
	}
	cudaFree(dhin);
	free(khout);
	return GPUTime;
}
//
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Hanning Window using CUDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Blackman Window using CUDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% Power Spectrum Stream Processing using CUDA GPUs %%%%%%%%%%%%%
//
double _10logAbs(vector<cuFloatComplex*> hdata, int dsize,int chunk,cuFloatComplex Vpp,cuFloatComplex divV, float logV)
{
	//
	struct timeval t1, t2;
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        //
	int grid1 = ceil(chunk/(float)threadsPerBlock);
	dimGrid.x = grid1;
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	//hdata.resize(dsize); // This must be done outside with data initialization/generation
	//hout.resize(dsize); // Might be a good idea to initialize outside of this function...
	cudaStream_t optisdr_streams[dsize];	
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	// Trying to Create Page-Locked std::vector - No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);//cudaHostAlloc(...);
		//cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Execution
	//
	for(int i = 0; i<dsize; i++)
	{
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	// Execution
	//	
	gettimeofday(&t1, 0);
	for(int i = 0; i<dsize; i++)
	{		
		//
		//
		_20log10fabs<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dout[i],chunk,Vpp,divV,logV);
		//
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&t2, 0);
	double GPUTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec); ///1000000.0;
	//
	for(int i = 0; i<dsize; i++)
	{		
		//
		cudaMemcpyAsync(hdata[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		//cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		//
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//
	}
	return GPUTime;
}
//
//
double _10logAbs(vector<cuFloatComplex*> hdata, int dsize,int chunk)
{
	//
	struct timeval t1, t2;
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        //
	int grid1 = ceil(chunk/(float)threadsPerBlock);
	dimGrid.x = grid1;
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //
	//hdata.resize(dsize); // This must be done outside with data initialization/generation
	//hout.resize(dsize); // Might be a good idea to initialize outside of this function...
	cudaStream_t optisdr_streams[dsize];	
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	// Trying to Create Page-Locked std::vector - No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);//cudaHostAlloc(...);
		//cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Execution
	//
	for(int i = 0; i<dsize; i++)
	{
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	// Execution
	//	
	gettimeofday(&t1, 0);
	for(int i = 0; i<dsize; i++)
	{		
		//
		//
		_10logabs<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dout[i],chunk);
		//
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&t2, 0);
	double GPUTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec); ///1000000.0;
	//
	for(int i = 0; i<dsize; i++)
	{		
		//
		cudaMemcpyAsync(hdata[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		//cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		//
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//
	}
	return GPUTime;
}
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% FFT Stream Processing using CUDA GPUs %%%%%%%%%%%%%%%%%%%%%%%%
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
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Batched FFT Stream processing using CUDA Streams and Pinned-Memory %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
double BatchedFFT(vector<cuFloatComplex*> hdata,int dsize,int chunk, int ftpoint)
{
	//
	struct timeval t1, t2;
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	int N_SIGS = chunk/ftpoint; // Chunk must be a multiple of ftpoint
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
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C, N_SIGS);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	// Execution
	//	
	gettimeofday(&t1, 0);
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&t2, 0);
	double GPUTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec); ///1000000.0;
	//
	for(int i = 0; i<dsize; i++)
	{		
		//
		cudaMemcpyAsync(hdata[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		//
	}
	//
	//
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		//cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	return GPUTime;
}
//
// Generic Stream for FFT
//
void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	//struct timeval t1, t2;
	//gettimeofday(&t1, 0);
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	int N_SIGS = chunk/ftpoint; // Chunk must be a multiple of ftpoint
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
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
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
//// XCorr Conjugate and Multiply
__global__ void xcorrmultiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	//
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//
	if(index < N)
	{
		outp[index] = cuCmulf(_cx[index],cuConjf(_cy[index]));
	}
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
double XCorrSP(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	struct timeval kernt1, kernt2;
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
	//cudaMalloc((void**)&dftrefsig,chunk*sizeof(cuFloatComplex));
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
	//
	gettimeofday(&kernt1, 0);
	//res = cufftExecC2C(plans[0],drefsig,dftrefsig,CUFFT_FORWARD);
	//if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		//complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dftrefsig,drout[i],chunk);
		xcorrmultiply<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],drefsig,drout[i],chunk);
		//
		res = cufftExecC2C(plans[i],drout[i],ddata[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Inverse transform fail.\n");}
		//
		// TODO: Try using the Block Size =128, i.e optisdrifftscale<<<chunk/128,128,0,optisdr_treams[i]>>>(...);
        	optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dout[i],ftpoint,chunk);
		//
		//cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&kernt2, 0);
	double GPUTime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec); ///1000000.0;
	for(int i = 0; i<dsize; i++)
	{		
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
	return GPUTime;
	//gettimeofday(&t2, 0);
	//double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	//printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//
//
double XCorrSP2(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	struct timeval kernt1, kernt2;
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
	//
	gettimeofday(&kernt1, 0);
	res = cufftExecC2C(plans[0],drefsig,dftrefsig,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		//complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dftrefsig,drout[i],chunk);
		xcorrmultiply<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dftrefsig,drout[i],chunk);
		//
		res = cufftExecC2C(plans[i],drout[i],ddata[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Inverse transform fail.\n");}
		//
		// TODO: Try using the Block Size =128, i.e optisdrifftscale<<<chunk/128,128,0,optisdr_treams[i]>>>(...);
        	optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dout[i],ftpoint,chunk);
		//
		//cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&kernt2, 0);
	double GPUTime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec); ///1000000.0;
	for(int i = 0; i<dsize; i++)
	{		
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
	return GPUTime;
	//gettimeofday(&t2, 0);
	//double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	//printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
double HilbertSP(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	struct timeval kernt1, kernt2;
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
	gettimeofday(&kernt1, 0);
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
		optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(drout[i],dout[i],ftpoint,chunk);
		//
		//cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	gettimeofday(&kernt2, 0);
	double GPUTime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec); ///1000000.0;
	//
	for(int i = 0; i<dsize; i++)
	{		
		//
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
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
	free(h);
	//
	return GPUTime;
}
//
cudaError_t cuda_main()
{
    // generate 16M random numbers on the host
    thrust::host_vector<int> h_vec(1 << 24);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
  
    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;
  
    // sort data on the device (805 Mkeys/sec on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());
  
    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  
    return cudaGetLastError();
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#endif

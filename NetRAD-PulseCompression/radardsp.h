#ifndef __RADARDSP__
#define __RADAR_DSP__
//
#include <unistd.h>
#include <iostream>
#include "qcustomplot.h"
#include "OptiSDRCuda.h"
#include "optisdrdevices.h"
#include "netradread.h"

using namespace std;

class RadarDSP
{

public:
	struct timeval t1, t2;
	vector< vector<cuFloatComplex*> > hout1; //,hout2;
	//cuFloatComplex* hout1;
	int numCPU;
	double k1time,k2time;
	double GPUcalTime;
	bool done;
	//cuFloatComplex* refsig =resizeVector(getReferenceSignal(rssize,ftpoint),ftpoint,(ftpoint/SIG_LEN)*chunk); // From 300 to chunk,
	//
	RadarDSP(NetRADRead *configs)
	{
		hout1.resize(configs->subchunk);
		//hout2.resize(configs->subchunks/2);
		//hout1 = getComplexEmpty(2*configs->dataLen);
		refsig = getComplexEmpty(2*configs->chunk);
		hdata0.resize(configs->dsize);
		hdata1.resize(configs->dsize);
		hdata2.resize(configs->dsize);
		numCPU = sysconf(_SC_NPROCESSORS_ONLN);
		gpu1done = false;
		gpu2done = false;
		done = false;
		k1time =0;
		k2time = 0;
		GPUcalTime = 0.0;
	}
	void zpad(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip)
	{
		int index = 2*skip;
		int iters = inlen/skip;
		//printf("%d",iters);
		for(int i=0; i<iters; i++)
		{
			for(int j=0; j<skip; j++)
			{
				outp[i*index + j] = inp[i*skip + j];
			}
		}
		//
	}
	
	void resize(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip)
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
	
	//
	void parProcess(NetRADRead *configs, OptiSDRDevices *devices,QCPColorMap *clMap)
	{
		// Graphics Vars
		double x, y, z;
		int yIndex2 = 0;
		int nx = 2*configs->ftpoint;
 		int ny = configs->chunk/configs->ftpoint;
		//
		cudaDeviceProp deviceProp;
		printf("[SDR_DSL_INFO]$ Chunking and Processing NetRAD Data. SubChunks:[%d], Chunks:[%d]\n",configs->subchunk,configs->dsize*configs->chunk);
		gettimeofday(&t1, 0);
		//
		//
		int M1=0, M2=0;
		//
		// Processing the Reference Signal - comp. sqrt-mean error - perform interference filter - add hamming Window
		printf("\n\n*******************\n 1. Reading Reference Signal :\n");
		cuFloatComplex *refin=getReferenceSignal("data/rref3sig2.dat","data/iref3sig2.dat",100);
		printf("\n 2. Interference Filtering :\n");
		_interferenceFilter(refin,5.0f,50.0f,100);
		printf("\n 3. Introducing Hamming Window to Ref. Sig. :\n");
		cuFloatComplex *fftref = hamming(refin,100,configs->ftpoint); // output equals ftpoint for Matched Filter
		printf("\n 4. FFTing the Reference Signal :\n");
		_fft(fftref,configs->ftpoint);
		//
		printf("\n 5. Zeropadding and extending Ref. Signal :\n");
		repMat(fftref,refsig,configs->ftpoint,configs->chunk,2); //Zero-Padding for XCorr - TODO: Move this to xcorr
		//
		free(refin); //Clean-up
		free(fftref); // Clean-up
		printf("\n 6. Done preparing the Ref. Signal...\n\n*****************\n\n");
		//
		int numCuDevs = 2;
		#pragma omp parallel for
		for(int cuda_dev = 0; cuda_dev<numCuDevs; cuda_dev++)
		{
			//
			cudaGetDevice(&cuda_dev); // Get Just the CUDA Device name
			cudaSetDevice(cuda_dev);
			//cudaGetDeviceProperties(&deviceProp, cuda_dev);		
			//printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_dev,deviceProp.name);
			for(int j = cuda_dev; j<configs->subchunk/(numCuDevs-cuda_dev); j++)
			{
				struct timeval k1t1, k1t2;
				gettimeofday(&k1t1, 0);
				int dfrom = configs->dsize*j;
				int dto = configs->dsize*j+configs->dsize;
				//
				hout1[j].resize(configs->dsize);
				//
				// Prepare Data for Processing: Chunk into independent partitions for Parallel Process
				//
				for(int i=dfrom; i<dto; i++)
				{
					int from = i*configs->chunk;
					int to = i*configs->chunk+configs->chunk;
					hdata0[i%configs->dsize] = getChunk(configs->vsNetRadSamples,from,to);
					hdata1[i%configs->dsize] = getComplexEmpty(configs->chunk);
					hout1[j][i%configs->dsize] = getComplexEmpty(2*configs->chunk);
				}
				//
				// Stream Processing the Hilbert Transform
				//
				double GPUTime = HilbertSP(hdata0,hdata1,configs->dsize,configs->chunk,configs->ftpoint);
				GPUcalTime+=GPUTime;
				//
				// Copy Data back to Host
				//
				for(int i=dfrom; i<dto; i++)
				{
					//
					int from = i*2*configs->chunk;
					int to = 2*configs->chunk;
					//
					hdata2[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
					zpad(hdata1[i%configs->dsize],hdata2[i%configs->dsize],configs->chunk,configs->ftpoint);
					//
				}
				//
				//
				GPUTime=XCorrSP(hdata2,refsig,hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
				GPUcalTime+=GPUTime;
				//
				//
				GPUTime=_10logAbs(hout1[j],configs->dsize,2*configs->chunk,make_cuFloatComplex(3.0f,0.0f),make_cuFloatComplex(pow(2,14),1.0f),20.0f);
				//GPUcalTime+=GPUTime;
				//
				gettimeofday(&k1t2, 0);
				k1time = k1time + (1000000.0*(k1t2.tv_sec-k1t1.tv_sec) + k1t2.tv_usec-k1t1.tv_usec)/1000000.0;			
				//
				for(int i=0; i<configs->dsize; i++)
				{
					//
					for(int yIndex=0; yIndex<ny; yIndex++)
					{
						#pragma omp parallel for
						for (int xIndex=0; xIndex<nx; xIndex++)
						{
							clMap->data()->cellToCoord(xIndex, yIndex2, &x, &y);
							//
							clMap->data()->setCell(xIndex, yIndex2, hout1[j][i][yIndex*nx + xIndex].x);
							//
						}
						yIndex2++;
					}				
					free(hdata1[i]);
					free(hout1[j][i]);
				}
				//
				if(j==((configs->subchunk/2)-1))
					gpu1done=true;
				//
			}
		}//
		//
		if(gpu1done)
		{
			free(refsig);
			//
			vector<cuFloatComplex*>().swap(hdata1);
			vector<cuFloatComplex*>().swap(hdata2);
			//
			vector<short>().swap(configs->vsNetRadSamples);
			//
			done = true;
			
		}
		//
		gettimeofday(&t2, 0);
		double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		//
		printf("[SDR_DSL_INFO]$ Time taken by GPUs Xcorr: [%f] .\n [SDR_DSL_INFO]$ Time Taken for Tiling, processing, and untiling = %f s. \n\n",k1time,time); //
		//
		//
		double GPUflops = configs->dataLen/(GPUcalTime*1e-6);
		double GPUflops2 = (configs->subchunk-(double)(M1+M2))*(configs->dsize*configs->chunk)/(GPUcalTime*1e-6);
		printf("[SDR_DSL_INFO]$ Overall XCorr Kernels Execution Time: [%f] seconds.\n\n [SDR_DSL_INFO]$ FLOPS: [%f or %f] MFLOPS.\n\n",GPUcalTime*1e-6,GPUflops*1e-6,GPUflops2*1e-6);
		cudaDeviceReset();
	}
	//
	void startProcess(NetRADRead *configs, OptiSDRDevices *devices,QCPColorMap *clMap)
	{
		// Graphics Vars
		double x, y, z;
		int yIndex2 = 0;
		int nx = 2*configs->ftpoint;
 		int ny = configs->chunk/configs->ftpoint;
		//
		cudaDeviceProp deviceProp;
		printf("[SDR_DSL_INFO]$ Chunking and Processing NetRAD Data. SubChunks:[%d], Chunks:[%d]\n",configs->subchunk,configs->dsize*configs->chunk);
		gettimeofday(&t1, 0);
		//devices->setCudaDevice(0);
		int cuda_device = 0;
		//
		int M1=0, M2=0;
		//
		cudaGetDevice(&cuda_device); // Get Just the CUDA Device name
		cudaSetDevice(cuda_device);
		cudaGetDeviceProperties(&deviceProp, cuda_device);
		//resize(getReferenceSignal(100,configs->ftpoint),refsig,configs->chunk,configs->ftpoint); // From 100 to chunk,
		//repMat(fftref,refsig,configs->ftpoint,configs->chunk,2);
		//
		// Processing the Reference Signal - comp. sqrt-mean error - perform interference filter - add hamming Window
		printf("\n\n*******************\n 1. Reading Reference Signal :\n");
		cuFloatComplex *refin=getRefSig("data/rref3sig2.dat","data/iref3sig2.dat",100);
		printf("\n 2. Interference Filtering :\n");
		_interferenceFilter(refin,5.0f,50.0f,100);
		printf("\n 3. Introducing Hamming Window to Ref. Sig. :\n");
		cuFloatComplex *fftref = hamming(refin,100,configs->ftpoint); // output equals ftpoint for Matched Filter
		printf("\n 4. FFTing the Reference Signal :\n");
		_fft(fftref,configs->ftpoint);
		//
		printf("\n 5. Zeropadding and extending Ref. Signal :\n");
		repMat(fftref,refsig,configs->ftpoint,configs->chunk,2); //Zero-Padding for XCorr - TODO: Move this to xcorr
		//
		free(refin); //Clean-up
		free(fftref); // Clean-up
		printf("\n 6. Done preparing the Ref. Signal...\n\n*****************\n\n");
		//		
		printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_device,deviceProp.name);
		for(int j = 0; j<configs->subchunk/2; j++)
		{
			struct timeval k1t1, k1t2;
			gettimeofday(&k1t1, 0);
			int dfrom = configs->dsize*j;
			int dto = configs->dsize*j+configs->dsize;
			//
			hout1[j].resize(configs->dsize);
			// Prepare Data for Processing: Chunk into independent partitions for Parallel Process
			//
			for(int i=dfrom; i<dto; i++)
			{
				//int to=i*chunk+chunk, from=i*chunk;
				int from = i*configs->chunk;
				int to = i*configs->chunk+configs->chunk;
				hdata0[i%configs->dsize] = getChunk(configs->vsNetRadSamples,from,to);
				hdata1[i%configs->dsize] = getComplexEmpty(configs->chunk);
				hout1[j][i%configs->dsize] = getComplexEmpty(2*configs->chunk);//(cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));//getComplexEmpty(2*configs->chunk);
				//
			}
			//
			// Stream Processing the Hilbert Transform
			//
			double GPUTime = HilbertSP(hdata0,hdata1,configs->dsize,configs->chunk,configs->ftpoint);
			GPUcalTime+=GPUTime;
			//printf("[SDR_DSL_INFO]$ Time: [%f]\n",GPUcalTime);
			//
			// Stream Processing the Cross Correlation
			//
			//vector<cuFloatComplex*>().swap(hdata0);
			//vector<short>().swap(configs->vsNetRadSamples);
			//hdata0.shrink_to_fit();
			//
			//
			// Copy Data back to Host
			//
			for(int i=dfrom; i<dto; i++)
			{
				//
				int from = i*2*configs->chunk;
				int to = 2*configs->chunk;
				//
				hdata2[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				zpad(hdata1[i%configs->dsize],hdata2[i%configs->dsize],configs->chunk,configs->ftpoint);
				//
				//hdata0[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				//
				//cout << "3. capacity of myvector: " << hdata0.capacity() << '\n';
				//copy(hdata2[i%configs->dsize] + 0,hdata2[i%configs->dsize] + to, hout1 + from);
			}
			//
			//
			GPUTime=XCorrSP(hdata2,refsig,hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
			GPUcalTime+=GPUTime;
			//printf("[SDR_DSL_INFO]$ Time: [%f]\n",GPUcalTime);
			//
			//GPUTime=_JSSHamming(hout1,configs->dsize,2*configs->chunk,2*configs->ftpoint,1);
			//GPUcalTime+=GPUTime;
			//
			//GPUTime=BatchedFFT(hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
			//GPUcalTime+=GPUTime;
			//
			GPUTime=_10logAbs(hout1[j],configs->dsize,2*configs->chunk,make_cuFloatComplex(3.0f,0.0f),make_cuFloatComplex(pow(2,14),1.0f),20.0f);
			//GPUcalTime+=GPUTime;
			//
			
			//
			gettimeofday(&k1t2, 0);
			k1time = k1time + (1000000.0*(k1t2.tv_sec-k1t1.tv_sec) + k1t2.tv_usec-k1t1.tv_usec)/1000000.0;			
			//
			for(int i=0; i<configs->dsize; i++)
			{
				//
				//
				//#pragma omp parallel for
				for(int yIndex=0; yIndex<ny; yIndex++)
				{
					#pragma omp parallel for
					for (int xIndex=0; xIndex<nx; xIndex++)
					{
						clMap->data()->cellToCoord(xIndex, yIndex2, &x, &y);
						//
						////z = cuCabsf(indata[yIndex*numx + xIndex]);
						//cuFloatComplex divV = make_cuFloatComplex(pow(2,14),1.0f);
						//cuFloatComplex multV = make_cuFloatComplex(3.0f,0.0f); //
						//z = 20.0f*log10f(cuCabsf(cuCdivf(cuCmulf(hout1[i][yIndex*nx + xIndex],multV),divV))); //
						////z = 20.0f*log10f(cuCabsf(hout[i][yIndex*nx + xIndex]));
						//z = cuCrealf(hout1[j][i][yIndex*nx + xIndex]);
						clMap->data()->setCell(xIndex, yIndex2, hout1[j][i][yIndex*nx + xIndex].x);
						//if(xIndex==0)
						//{
						//	printf("%f ",z);
						//}
					}
					yIndex2++;
				}				
				free(hdata1[i]);
				free(hout1[j][i]);
			}
			//swap(hout1[j],hdata0);
			//
			/*for(int i=dfrom; i<dto; i++)
			{
				//
				//int from = i*configs->chunk;
				//int to = configs->chunk;
				//
				//free(hdata0[i%configs->dsize]);
				free(hdata1[i]);
				free(hout1[j][i]);
			}*/
			//
			if(j==((configs->subchunk/2)-1))
				gpu1done=true;
			//
		}
		//
		if(gpu1done)
		{
			//free(refsig);
			//vector<cuFloatComplex*>().swap(hout1);
			vector<cuFloatComplex*>().swap(hdata1);
			vector<cuFloatComplex*>().swap(hdata2);
			//vector<short>().swap(configs->vsNetRadSamples);
			//done = true;
		}
		cudaSetDevice(cuda_device+1);
		cudaGetDeviceProperties(&deviceProp, cuda_device+1);		
		printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_device+1,deviceProp.name);
		//
		hdata1.resize(configs->dsize);
		hdata2.resize(configs->dsize);
		for(int j = configs->subchunk/2; j<configs->subchunk; j++)
		{
			struct timeval k2t1, k2t2;
			gettimeofday(&k2t1, 0);
			int dfrom = configs->dsize*j;
			int dto = configs->dsize*j+configs->dsize;
			//
			hout1[j].resize(configs->dsize);
			//
			// Prepare Data for Processing: Chunk into independent partitions for Parallel Process
			for(int i=dfrom; i<dto; i++)
			{
				//int to=i*chunk+chunk, from=i*chunk;
				int from = i*configs->chunk;
				int to = i*configs->chunk+configs->chunk;
				hdata0[i%configs->dsize] = getChunk(configs->vsNetRadSamples,from,to);
				hdata1[i%configs->dsize] = getComplexEmpty(configs->chunk);
				hout1[j][i%configs->dsize] = getComplexEmpty(2*configs->chunk); //(cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));//
				//
			}
			//
			//
			// Stream Processing the Hilbert Transform
			//
			double GPUTime = HilbertSP(hdata0,hdata1,configs->dsize,configs->chunk,configs->ftpoint);			
			GPUcalTime+=GPUTime;
			//
			// Copy Data back to Host
			//
			for(int i=dfrom; i<dto; i++)
			{
				//
				int from = i*2*configs->chunk;
				int to = 2*configs->chunk;
				//
				hdata2[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				zpad(hdata1[i%configs->dsize],hdata2[i%configs->dsize],configs->chunk,configs->ftpoint);
				//
				//
			}
			GPUTime = XCorrSP(hdata2,refsig,hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
			GPUcalTime+=GPUTime;
			//printf("[SDR_DSL_INFO]$ Time: [%f]\n",GPUcalTime);
			//
			//GPUTime=_JSSHamming(hout1,configs->dsize,2*configs->chunk,2*configs->ftpoint,1);
			//GPUcalTime+=GPUTime;
			//
			//
			//
			//GPUTime=BatchedFFT(hout1,configs->dsize,2*configs->chunk,2*configs->ftpoint);
			//GPUcalTime+=GPUTime;
			//
			//
			GPUTime=_10logAbs(hout1[j],configs->dsize,2*configs->chunk,make_cuFloatComplex(3.0f,0.0f),make_cuFloatComplex(pow(2,14),1.0f),20.0f);
			//
			//GPUTime=_10logAbs(hout1,configs->dsize,2*configs->chunk);
			//GPUcalTime+=GPUTime;
			//			
			gettimeofday(&k2t2, 0);
			k2time = k2time + (1000000.0*(k2t2.tv_sec-k2t1.tv_sec) + k2t2.tv_usec-k2t1.tv_usec)/1000000.0;
			//
			//swap(hout1[j],hdata0); //hout1[j] = hdata0;
			//
			//
			//#pragma omp parallel for
			for(int i=0; i<configs->dsize; i++)
			{
				//
				//
				//#pragma omp parallel for
				for(int yIndex=0; yIndex<ny; ++yIndex)
				{
					#pragma omp parallel for
					for (int xIndex=0; xIndex<nx; ++xIndex)
					{
						clMap->data()->cellToCoord(xIndex, yIndex2, &x, &y);
						//
						////z = cuCabsf(indata[yIndex*numx + xIndex]);
						//cuFloatComplex divV = make_cuFloatComplex(pow(2,14),1.0f);
						//cuFloatComplex multV = make_cuFloatComplex(3.0f,0.0f);
						//z = 20.0f*log10f(cuCabsf(cuCdivf(cuCmulf(hout2[i][yIndex*nx + xIndex],multV),divV)));
						//z = cuCrealf(hout1[j][i][yIndex*nx + xIndex]);
						//z = hout1[j][i][yIndex*nx + xIndex].x;
						////z = 20.0f*log10f(cuCabsf(hout2[i][yIndex*nx + xIndex])); //
						clMap->data()->setCell(xIndex, yIndex2, hout1[j][i][yIndex*nx + xIndex].x);
						//if(xIndex==0)
						//{
						//	printf("%f ",z);
						//}
					}
					yIndex2++;
				}				
				free(hdata1[i]);
				free(hout1[j][i]);
			}
			//
			//
			/*for(int i=0; i<configs->dsize; i++)
			{
				//
				//free(hdata0[i%configs->dsize]);
				free(hdata1[i]);
				free(hout1[j][i]);
			}
			*/
			//
			//
			if(j==((configs->subchunk/2)-1))
				gpu2done=true;
			//
		}
		if(gpu1done && gpu2done)
		{
			free(refsig);
			//for(int t=0;t<configs->dsize;t++)
			//{
			//	free(hdata2[t]);
			//}
			//vector<cuFloatComplex*>().swap(hdata0);
			vector<cuFloatComplex*>().swap(hdata1);
			vector<cuFloatComplex*>().swap(hdata2);
			//vector<cuFloatComplex*>().swap(hout1);
			//vector<cuFloatComplex*>().swap(hout2);
			vector<short>().swap(configs->vsNetRadSamples);
			//TestStuff();
			
		}
		//hdata0.shrink_to_fit();
		gettimeofday(&t2, 0);
		double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
		printf("[SDR_DSL_INFO]$ Time taken by GPU 1 Xcorr: [%f] .\n [SDR_DSL_INFO]$ Time taken by GPU 2 Xcorr: [%f] .\n [SDR_DSL_INFO]$ Overall time for Xcorr: [%f] .\n [SDR_DSL_INFO]$ Time Taken for Tiling, processing, and untiling = %f s. \n\n",k1time,k2time,k1time+k2time,time); // Get rit of the untiling part, cause degradation.
		//
		//
		double GPUflops = configs->dataLen/(GPUcalTime*1e-6);
		double GPUflops2 = (configs->subchunk-(double)(M1+M2))*(configs->dsize*configs->chunk)/(GPUcalTime*1e-6);
		printf("[SDR_DSL_INFO]$ Overall XCorr Kernels Execution Time: [%f] seconds.\n\n [SDR_DSL_INFO]$ FLOPS: [%f or %f] MFLOPS.\n\n",GPUcalTime*1e-6,GPUflops*1e-6,GPUflops2*1e-6);
		cudaDeviceReset();
	}
private:
	vector<cuFloatComplex*> hdata0,hdata1,hdata2;
	cuFloatComplex* refsig;
	bool gpu1done,gpu2done;
	cuFloatComplex* getRefSig(string fnamer, string fnamei, int _dlen)
	{
		cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
		//
		vector<float> rSamples,iSamples;
		readNetRadSamps(fnamer,_dlen,rSamples);
		readNetRadSamps(fnamei,_dlen,iSamples);
		//
		for(int i = 0; i< _dlen; i++)
		{
			refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
		}
		//printf("\n\n%f\n\n\n",refsig[_dlen-1]);
		return refsig;
	}
	//
	void readNetRadSamps(string strFilename, unsigned int nsamples, vector<float> &vsSamples)
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
};
#endif

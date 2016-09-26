
	void cuNetRADProcessor(NetRADRead *configs, OptiSDRDevices *devices)
	{
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
		printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_device,deviceProp.name);
		resize(getReferenceSignal(100,configs->ftpoint),refsig,configs->chunk,configs->ftpoint); // From 100 to chunk,
		/*printf("\n\n[ ");
		for(int t=0;t<100;t++)
		{
			printf("%f,",cuCrealf(refsig[t]));
		}
		printf(" ]\n\n");
		*/
		//
		for(int j = 0; j<configs->subchunk/2-M1; j++)
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
				hout1[j][i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));//getComplexEmpty(2*configs->chunk);
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
			GPUTime=BatchedFFT(hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
			//GPUcalTime+=GPUTime;
			//
			GPUTime=_10logAbs(hout1[j],configs->dsize,2*configs->chunk);
			//GPUcalTime+=GPUTime;
			//
			//
			gettimeofday(&k1t2, 0);
			k1time = k1time + (1000000.0*(k1t2.tv_sec-k1t1.tv_sec) + k1t2.tv_usec-k1t1.tv_usec)/1000000.0;			
			//
			/*for(int i=dfrom; i<dto; i++)
			{
				//
				int from = i*2*configs->chunk;
				int to = 2*configs->chunk;
				//
				//hdata2[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				//zpad(hdata1[i%configs->dsize],hdata2[i%configs->dsize],configs->chunk,configs->ftpoint);
				//
				//hdata0[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				//
				//cout << "3. capacity of myvector: " << hdata0.capacity() << '\n';
				copy(hdata0[i%configs->dsize] + 0,hdata0[i%configs->dsize] + to, hout1 + from);
			}*/
			//swap(hout1[j],hdata0);
			//
			for(int i=dfrom; i<dto; i++)
			{
				//
				//int from = i*configs->chunk;
				//int to = configs->chunk;
				//
				//free(hdata0[i%configs->dsize]);
				free(hdata1[i%configs->dsize]);
				//free(hdata2[i%configs->dsize]);
			}
			
			//
			////printf("\nTesting Here ...After Xcorr...\n");
			//
			if(j==((configs->subchunk/2)-1))
				gpu1done=true;
			//
		}
		//
		if(gpu1done)
		{
			//free(refsig);
			//vector<cuFloatComplex*>().swap(hdata0);
			vector<cuFloatComplex*>().swap(hdata1);
			vector<cuFloatComplex*>().swap(hdata2);
			//vector<short>().swap(configs->vsNetRadSamples);
		}
		cudaSetDevice(cuda_device+1);
		cudaGetDeviceProperties(&deviceProp, cuda_device+1);		
		printf("[SDR_DSL_INFO]$ Device ID :[%d] Name:[%s]\n",cuda_device+1,deviceProp.name);
		//
		hdata1.resize(configs->dsize);
		hdata2.resize(configs->dsize);
		for(int j = configs->subchunk/2; j<configs->subchunk-M2; j++)
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
				hout1[j][i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));//getComplexEmpty(2*configs->chunk);
				//
			}
			//
			//
			// Stream Processing the Hilbert Transform
			//
			double GPUTime = HilbertSP(hdata0,hdata1,configs->dsize,configs->chunk,configs->ftpoint);			
			GPUcalTime+=GPUTime;
			//printf("[SDR_DSL_INFO]$ Time: [%f]\n",GPUcalTime);
			//
			// Stream Processing the Cross Correlation
			//
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
			GPUTime = XCorrSP(hdata2,refsig,hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
			GPUcalTime+=GPUTime;
			//printf("[SDR_DSL_INFO]$ Time: [%f]\n",GPUcalTime);
			//
			//
			GPUTime=BatchedFFT(hout1[j],configs->dsize,2*configs->chunk,2*configs->ftpoint);
			//GPUcalTime+=GPUTime;
			//
			//
			GPUTime=_10logAbs(hout1[j],configs->dsize,2*configs->chunk);
			//GPUcalTime+=GPUTime;
			//			
			gettimeofday(&k2t2, 0);
			k2time = k2time + (1000000.0*(k2t2.tv_sec-k2t1.tv_sec) + k2t2.tv_usec-k2t1.tv_usec)/1000000.0;
			//
			//swap(hout1[j],hdata0); //hout1[j] = hdata0;
			/*
			for(int i=dfrom; i<dto; i++)
			{
				//
				int from = i*2*configs->chunk;
				int to = 2*configs->chunk;
				//
				//hdata2[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				//zpad(hdata1[i%configs->dsize],hdata2[i%configs->dsize],configs->chunk,configs->ftpoint);
				//
				//hdata0[i%configs->dsize] = (cuFloatComplex*)malloc(2*configs->chunk*sizeof(cuFloatComplex));
				//
				//cout << "3. capacity of myvector: " << hdata0.capacity() << '\n';
				copy(hdata0[i%configs->dsize] + 0,hdata0[i%configs->dsize] + to, hout1 + from);
			}*/
			//
			for(int i=dfrom; i<dto; i++)
			{
				//
				int from = i*configs->chunk;
				int to = configs->chunk;
				//
				//free(hdata0[i%configs->dsize]);
				free(hdata1[i%configs->dsize]);
				//free(hdata2[i%configs->dsize]);
			}
			
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

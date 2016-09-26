


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

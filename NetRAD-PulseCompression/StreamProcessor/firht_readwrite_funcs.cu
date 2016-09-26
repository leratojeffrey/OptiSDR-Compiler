//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef _CUTESTO_H_
#define _CUTESTO_H_

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//	cutWriteFilef( "./data/input.dat", h_idata, x_len, 0.0)

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

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool compareF (float *reference, float *data, const unsigned int len, const float epsilon)
{
    float diff;
    unsigned int i=0, err=0;
    bool result = true;
    
    for (i=0; i<len; ++i) 
    {
// Absolute
//	diff = reference[i] - data[i];
//	bool comp = (diff <= epsilon) && (diff >= -epsilon);
// Relative
	diff = (reference[i] - data[i] ) / reference[i];
	bool comp = (diff <= epsilon) && (diff >= -epsilon);
	if ( !comp ) {
	    err++;
#ifdef _DEBUG
	    printf("[SDR_DSL_INFO]$ ErrNo.: %3d at %3d Ref:%.7e \tData:%.7e \trDiff:%.7e\n", err, i, reference[i], data[i], diff);
#endif
	}
	result &= comp;
    }
    if ( !result ) {printf("[SDR_DSL_INFO]$ %d Errors of %d samples\n", err, len);}
    return (result) ? true : false;

}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//Complex Data File Write

void writeFileF(const char *fpath,cuFloatComplex *xdata, const unsigned int len)
{
    printf("[SDR_DSL_INFO]$ Output file: %s\n", fpath);
    FILE *fo;

    unsigned int i=0;

    if ( (fo = fopen(fpath, "w")) == NULL) {printf("[SDR_DSL_INFO]$ IO Error\n");}

    for (i=0; i<len; ++i)
    {
	if((fprintf(fo,"%.7e + %.7ei\n",cuCrealf(xdata[i]),cuCimagf(xdata[i]))) <= 0 )
	{
		printf("[SDR_DSL_INFO]$ File write Error.\n");
		fclose(fo);
	}
    }

    fclose(fo);
    //return(CUTTrue);
}
//
void ddisp(cuFloatComplex *xdata, const unsigned int len)
{
    unsigned int i=0;

    printf("[");

    for (i=0; i<len; ++i)
    {
	printf("%f + %fi",cuCrealf(xdata[i]),cuCimagf(xdata[i]));
	if(i!=(len-1))
		printf(", ");
    }
    printf("]\n");
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#endif // _CUTESTO_H_
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

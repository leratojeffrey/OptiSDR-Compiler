//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: FIR Filter Hilbert Transform Approximator Kernels %%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%
/*#ifndef _THEFIR_KERNEL_H_
#define _THEFIR_KERNEL_H_
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Defining the emu Keyword
#ifdef emu
 #include <stdio.h>
#endif

#define SDATA( index) CUT_BANK_CHECKER(sdata, index)

#define MAXTHREADS 512

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%% Declaring Texture Memory for Holding COefficiends %%%%%%%%%%%%%%%%%%%%
//%%%%%%% Coefficients texture reference // no conversion %%%%%%%%%%%%%%%%%%%%%
//%%%%%%%% Texture from cuda array - use tex1D() to fetch %%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%% texture<float, 1,cudaReadModeElementType> RemezCoeffs; %%%%%%%%%%%

texture<float, 1,cudaReadModeElementType> RemezCoeffs;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%% FIR Filter Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void transposed_fir(float *d_VAL_A,float* Output_Data, const unsigned int TapOrder,const unsigned int J)
{
    // Shared memory,the size is determined by the host application
    extern  __shared__  float OutputData[];
    // Thread index - We be using this - Threads for Computing a Single FIR Filter Operation
    const unsigned int tid = threadIdx.x;
    // Actual Nunber of Threads per Block
    const unsigned int NT = blockDim.x;
    // Block index - May not be necessary
    const unsigned int bid = blockIdx.x;
    // Number of blocks - Required if we need to Pipeline or We have TapLength > 1024 (highly unnecessary)
    const unsigned int NB = gridDim.x;
    double sum = 0; // Sum is used to hold convolution data...
    unsigned int i = 0, j = 0; // Lets initialize indexers i and j globally in this functions... We'll see why below

//#ifdef emu
//    printf("Testing If we get here...!");
//#endif
    // Perform Sum and Shift in Nested for(...) Loops
    for (j=0; j<J; j++)
    {
        for (i=0; i<TapOrder; ++i)
        {
            sum += (float)((float) tex1D(RemezCoeffs, (unsigned int)i ) * d_VAL_A[(unsigned int)(i+tid+TapOrder*j + bid*J*TapOrder)]);
        }
        __syncthreads();
        OutputData[tid+TapOrder*j] = (float)sum;
        __syncthreads();
        sum = (double)0.0;
    }
    // Do the last sample; No. of proper FIR outputs is N-M+1
    if (tid == 0) 
    {
        for (i=0; i<TapOrder; i++)
        {
            sum += (float)((float)tex1D(RemezCoeffs,(unsigned int)i) * d_VAL_A[(unsigned int)(i+tid+TapOrder*J + bid*J*TapOrder)]);
        }
    }
    __syncthreads();
    if (tid == 0) //Insert into Shared Memory
    {
        OutputData[tid+TapOrder*J] = (float)sum;
    }
    __syncthreads();
    
    for (j=0; j<J; j++) //After we sum and shift, Lets insert data into the Global Memory, Why?
    {
        Output_Data[tid+j*TapOrder + bid*J*TapOrder] = (float)OutputData[tid+J*TapOrder];
    }
    __syncthreads();
    
    if (tid == 0)
    {
        Output_Data[tid+J*TapOrder + bid*J*TapOrder] = (float)OutputData[tid+J*TapOrder];
    }
}

#endif*/

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: FIR Filter Hilbert Transform Approximator %%%%%%%%%%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Include Cuda run-time and inline Libraries
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>

// Helper libraries that are common to CUDA SDK samples
//#include <helper_cuda.h>
//#include <helper_functions.h>
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%% Including the Kernel: FIR Filter Calculator %%%%%%%%%%%%%%%%%%%%%%%%%%
#include "firht_kernel.cu"

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Data configuration and Memory Allocation %%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//Total number of input vector pairs; arbitrary
const int VECTOR_N  = 256;

//Number of elements per vector; arbitrary, 
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
const int ELEMENT_N = 4096;

//Total number of data elements
const int DATA_N    = VECTOR_N * ELEMENT_N;
const int DATA_SZ   = DATA_N * sizeof(float);
const int RESULT_SZ = VECTOR_N  * sizeof(float);

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%% Define local points to memory blocks
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
float *hDataInA, *hDataInB, *h_C_CPU, *h_C_GPU;
float *dDataInA, *dDataInB, *dDataInC;
double delta, ref, sum_delta, sum_ref, L1norm;
unsigned int hTimer;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%% Lets See if We can Borrow from nVidia Helper Functions %%%%%%%%%%%
inline int _ConvertSMVer2Cores(int major, int minor)
{    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] ={
        {0x10,8}, // Tesla Generation (SM 1.0) G80 class
        {0x11,8}, // Tesla Generation (SM 1.1) G8x class
        {0x12,8}, // Tesla Generation (SM 1.2) G9x class
        {0x13,8}, // Tesla Generation (SM 1.3) GT200 class
        {0x20,32}, // Fermi Generation (SM 2.0) GF100 class
        {0x21,48}, // Fermi Generation (SM 2.1) GF10x class
        {0x30,192}, // Kepler Generation (SM 3.0) GK10x class
        {0x35,192}, // Kepler Generation (SM 3.5) GK11x class
        {-1,-1}
    };
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);

    return nGpuArchCoresPerSM[7].Cores;
}
//%%%%%% Good We Can .....! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Safe shutdown routine: Adopted from EEE40xx Prac02 %%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void do_shutdown( int argc, char **argv)
{
    printf("Shutting down...\n");
        cutilSafeCall( cudaFree(dDataInC) );
        cutilSafeCall( cudaFree(dDataInB));
        cutilSafeCall( cudaFree(dDataInA));
        free(h_C_GPU);
        free(h_C_CPU);
        free(hDataInB);
        free(hDataInA);
        // This may not be necessary unless we need to time something
        cutilCheckError(cutDeleteTimer(hTimer) );

        cudaThreadExit();
        cutilExit(argc, argv);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Declaring Variables and Functions for Multiple Copy Streaming %%
#define STREAM_COUNT 4

// Uncomment to simulate data source/sink IO times
//#define SIMULATE_IO

int *h_data_source;
int *h_data_sink;

int *h_data_in[STREAM_COUNT];
int *d_data_in[STREAM_COUNT];

int *h_data_out[STREAM_COUNT];
int *d_data_out[STREAM_COUNT];


cudaEvent_t cycleDone[STREAM_COUNT];
cudaStream_t stream[STREAM_COUNT];

cudaEvent_t start, stop;

int N = 1 << 22;
int nreps = 10;                 // number of times each experiment is repeated
int inner_reps = 5;

int memsize;

dim3 block(512);
dim3 grid;

int thread_blocks;

float processWithStreams(int streams_used);
void init();
bool test();

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int main(int argc, char **argv)
{

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Lets try getting Device Information First
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for(int i=0; i< count; i++) 
    {
        //Get Device Properties using cudaGetDevieProperties(...)
        cudaGetDeviceProperties(&prop,i);
        //Displaying Device Information

        printf( "[SDR_DSL]***General Information for device %d***[SDR_DSL]\n", i );
        // Printing Device Name - i.e. GeForce GTX ...
        printf( "[SDR_DSL_INFO]$ Name:%s\n", prop.name );
        // Print Compute Capability
        printf( "[SDR_DSL_INFO]$ Compute capability:%d.%d\n", prop.major, prop.minor );
        //Print Clock Rate
        printf( "[SDR_DSL_INFO]$ Clock rate: %d\n", prop.clockRate );
        //Can we simultaneously perform a cudaMemcpy() and kernel execution
        printf( "[SDR_DSL_INFO]$ Device copy overlap:" );
        if(prop.deviceOverlap)
        {
            printf( "Enabled\n" );
        }
        else
        {
            printf( "Disabled\n" );
        }
        //Print Kernel Execution Time
        printf( "[SDR_DSL_INFO]$ Kernel execition timeout :" );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
    
        //Print Memory Information
        printf( "[SDR_DSL]***Memory Information for device %d***[SDR_DSL]\n", i );
        //Total Global Memory
        printf( "[SDR_DSL_INFO]$ Total global mem:%ld\n", prop.totalGlobalMem );
        // Total Constant Memory
        printf( "[SDR_DSL_INFO]$ Total constant Mem:%ld\n", prop.totalConstMem );
        // Total Memory Pitch
        printf( "[SDR_DSL_INFO]$ Max mem pitch:%ld\n", prop.memPitch );
        // Texture Alignment
        printf( "[SDR_DSL_INFO]$ Texture Alignment:%ld\n", prop.textureAlignment );
        
        printf( "[SDR_DSL]***MP Information for device %d***[SDR_DSL]\n", i );
        // Printing MultiProcessor Count - 
        printf( "[SDR_DSL_INFO]$ Multiprocessor count:%d\n",prop.multiProcessorCount );
        // Print Shared Memory
        printf( "[SDR_DSL_INFO]$ Shared mem per mp:%ld\n", prop.sharedMemPerBlock );
        // Print Registers Per Block
        printf( "[SDR_DSL_INFO]$ Registers per mp:%d\n", prop.regsPerBlock );
        // Print Threads Per Warp
        printf( "[SDR_DSL_INFO]$ Threads in warp:%d\n", prop.warpSize );
        // Display Maximum Thread per Block
        printf( "[SDR_DSL_INFO]$ Max threads per block:%d\n",prop.maxThreadsPerBlock );
        // Displaying Maximum Dimensions for each Thread
        printf( "[SDR_DSL_INFO]$ Max thread dimensions:(%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2] );
        // Print Maximum Grid Dimensions
        printf( "[SDR_DSL_INFO]$ Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );

        printf( "\n" );
    }
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%% Fill a cudaDeviceProp structure with the properties %%%%%%%%%%%%
    //%%%%%%%%% we need our device to have %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cudaDeviceProp deviceProp;
    //int newdev;
    //cudaGetDevice( &newdev );
    //printf( "ID of current CUDA device: %d\n", newdev );
    //memset( &newprop, 0, sizeof( cudaDeviceProp ) );
    //newprop.major = 1;
    //newprop.minor = 3;
    //cudaChooseDevice( &newdev, &newprop );
    //printf( "ID of CUDA device closest to revision 1.3: %d\n",newdev);
    //cudaSetDevice(newdev); // This maybe overidden by th cuSetDevice(...) below
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //int i; // counter

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%% Use command-line specified CUDA device, %%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%% otherwise use device with highest Gflops/s %%%%%%%%%%%%%%%%
    int cuda_device = 0;    
    // Get device from Command Line Arguments
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
    {
        cutilDeviceInit(argc, argv);
    }
    else// Otherwise pick the device with the highest Gflops/s
    {
        cuda_device = cutGetMaxGflopsDeviceId();
        cudaSetDevice(cuda_device);
        cudaGetDeviceProperties(&deviceProp, cuda_device);
        printf("[SDR_DSL_INFO]$ Using CUDA device [%d]: %s\n", cuda_device, deviceProp.name);
    }
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%% My Main code starts here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    float scale_factor;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%% Checking Number of Cores and Scaling Down for < 32 Cores %%%%%%
    scale_factor = max((32.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);
    N = (int)((float)N / scale_factor);

    printf("[SDR_DSL_INFO]$ Device name: %s\n", deviceProp.name);
    printf("[SDR_DSL_INFO]$ CUDA Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor,
           deviceProp.multiProcessorCount);
    printf("[SDR_DSL_INFO]$ Scale_Factor = %.2f\n", 1.0f/scale_factor);
    printf("[SDR_DSL_INFO]$ array_size   = %d\n\n", N);

    memsize = N * sizeof(int);

    thread_blocks = N / block.x;

    grid.x = thread_blocks % 65535;
    grid.y = (thread_blocks / 65535 + 1);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%% Allocating GPU and Host Memory %%%%%%%%%%%%%%%%% 
    //%%%%%%%%%%% And Creating Stream - Good for real-time processing %%%%%%%%
    h_data_source = (int *) malloc(memsize); // Data Source memory in CPU
    h_data_sink = (int *) malloc(memsize); //Data Sink Memory in CPU
    
    // Allocate GPU and Host Memory
    // We have STREAM_COUNT of 4 so we allocate 4 Streams for Device and Host
    for (int i =0; i<STREAM_COUNT; ++i)
    {

        cudaHostAlloc(&h_data_in[i], memsize,cudaHostAllocDefault);
        cudaMalloc(&d_data_in[i], memsize);

        cudaHostAlloc(&h_data_out[i], memsize,cudaHostAllocDefault);
        cudaMalloc(&d_data_out[i], memsize);

        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&cycleDone[i]);

        cudaEventRecord(cycleDone[i], stream[i]);
    }// End of Stream and Memory Allocation
    // Creating Timers/Time Stamps for Start and Stop times
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    init(); // Host memory initialization using memcpy
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%% Initial Kernel Calling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    incKernel<<<grid, block>>>(d_data_out[0], d_data_in[0], N, inner_reps);
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%% Recording Data Copies h2d & d2h, and Kernel Exec. Time %%%%%%%%
    cudaEventRecord(start,0); // Start Timer for host to dev copy
    cudaMemcpyAsync(d_data_in[0], h_data_in[0], memsize,cudaMemcpyHostToDevice,0); // Do host to Device Async Copy
    cudaEventRecord(stop,0); // stop Timer for H 2 D Async. Copy
    cudaEventSynchronize(stop); // Synchronize Stop timer

    float memcpy_h2d_time; // H2D Copy Time Varaible
    cudaEventElapsedTime(&memcpy_h2d_time, start, stop); // Record H2D Copy time

    cudaEventRecord(start,0);// Start Timer for dev to host copy
    cudaMemcpyAsync(h_data_out[0], d_data_out[0], memsize,cudaMemcpyDeviceToHost, 0); // Do dev to host Async Copy
    cudaEventRecord(stop,0); // stop Timer for D 2 H Async. Copy
    cudaEventSynchronize(stop); // Synch Stop Timer

    float memcpy_d2h_time; //D2H Copy Time Variable
    cudaEventElapsedTime(&memcpy_d2h_time, start, stop);  // Record D2H Copy time

    cudaEventRecord(start,0); // Start Timer for Kernel Execution
    incKernel<<<grid, block,0,0>>>(d_data_out[0], d_data_in[0], N, inner_reps); // Call Kernel
    cudaEventRecord(stop,0); //Stop Timer for Kernel Execution
    cudaEventSynchronize(stop); // Synch Stop timer

    float kernel_time; // Kernel Time Variable
    cudaEventElapsedTime(&kernel_time, start, stop);// Record Kernel exec. time
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%% Reporting Mem. Copy and Execution Times %%%%%%%%%%%%
    printf("\n");
    printf("[SDR_DSL_REP]$ Relevant properties of this CUDA device\n");
    printf("[SDR_DSL_REP]$ (%s) Can overlap one CPU<>GPU data transfer with GPU kernel execution (device property \"deviceOverlap\")\n", deviceProp.deviceOverlap ? "X" : " ");

    printf("[SDR_DSL_REP]$ (%s) Can overlap two CPU<>GPU data transfers with GPU kernel execution\n (Compute Capability >= 2.0 AND (Tesla product OR Quadro 4000/5000/6000/K5000)\n",
           (deviceProp.major >= 2 && deviceProp.asyncEngineCount > 1) ? "X" : " ");

    printf("\n");
    printf("[SDR_DSL_REP]$ Measured timings (throughput):\n");
    printf("[SDR_DSL_REP]$ Memcpy host to device\t: %f ms (%f GB/s)\n",memcpy_h2d_time, (memsize * 1e-6)/ memcpy_h2d_time);
    printf("[SDR_DSL_REP]$ Memcpy device to host\t: %f ms (%f GB/s)\n",memcpy_d2h_time, (memsize * 1e-6)/ memcpy_d2h_time);
    printf("[SDR_DSL_REP]$ Kernel\t\t\t: %f ms (%f GB/s)\n",kernel_time, (inner_reps *memsize * 2e-6)/ kernel_time);

    printf("\n");
    printf("[SDR_DSL_REP]$ Theoretical limits for speedup gained from overlapped data transfers:\n");
    printf("[SDR_DSL_REP]$ No overlap at all (transfer-kernel-transfer): %f ms \n",memcpy_h2d_time + memcpy_d2h_time + kernel_time);
    printf("[SDR_DSL_REP]$ Compute can overlap with one transfer: %f ms\n",max((memcpy_h2d_time + memcpy_d2h_time), kernel_time));
    printf("[SDR_DSL_REP]$ Compute can overlap with both data transfers: %f ms\n",max(max(memcpy_h2d_time,memcpy_d2h_time), kernel_time));    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%% Performing Pipelined Work %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Process pipelined work
    float serial_time = processWithStreams(1);
    float overlap_time = processWithStreams(STREAM_COUNT);

    printf("\n[SDR_DSL_REP]$ Average measured timings over %d repetitions:\n", nreps);
    printf("[SDR_DSL_REP]$ Avg. time when execution fully serialized\t: %f ms\n",serial_time / nreps);
    printf("[SDR_DSL_REP]$ Avg. time when overlapped using %d streams\t: %f ms\n",STREAM_COUNT, overlap_time / nreps);
    printf("[SDR_DSL_REP]$ Avg. speedup gained (serialized - overlapped)\t: %f ms\n",(serial_time - overlap_time) / nreps);

    printf("\n[SDR_DSL_REP]$ Measured throughput:\n");
    printf("[SDR_DSL_REP]$ Fully serialized execution\t\t: %f GB/s\n",(nreps * (memsize * 2e-6))/ serial_time);
    printf("[SDR_DSL_REP]$ Overlapped using %d streams\t\t: %f GB/s\n",STREAM_COUNT, (nreps * (memsize * 2e-6))/ overlap_time);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%% Results Verification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Verify the results, we will use the results for final output
    bool bResults = test();
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%% Shutting Down and Clearing Memory %%%%%%%%%%%%%%%%%%%
    //do_shutdown(argc,argv);
    // Free resources
    free(h_data_source);
    free(h_data_sink);

    for (int i =0; i<STREAM_COUNT; ++i)
    {
        cudaFreeHost(h_data_in[i]);
        cudaFree(d_data_in[i]);

        cudaFreeHost(h_data_out[i]);
        cudaFree(d_data_out[i]);

        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(cycleDone[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    // Test result
    exit(bResults ? EXIT_SUCCESS : EXIT_FAILURE);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Results Verification Helper Functions %%%%%%%%%%%%%%%%%%%%%%%
bool test()
{
    bool passed = true;
    for (int j =0; j<STREAM_COUNT; ++j)
    {
        for (int i =0; i<N; ++i)
        {
            passed &= (h_data_out[j][i] == 1);
        }
    }
    return passed;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Memory Updating Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
float processWithStreams(int streams_used)
{
    int current_stream = 0;
    float time;
    // Do processing in a loop:
    // Note: All memory commands are processed in the order  they are issued,
    // independent of the stream they are enqueued in. Hence the pattern by
    // which the copy and kernel commands are enqueued in the stream
    // has an influence on the achieved overlap.
    cudaEventRecord(start, 0);
    for (int i=0; i<nreps; ++i)
    {
        int next_stream = (current_stream + 1) % streams_used;
#ifdef SIMULATE_IO
        // Store the result
        memcpy(h_data_sink, h_data_out[current_stream],memsize);
        // Read new input
        memcpy(h_data_in[next_stream], h_data_source, memsize);
#endif
        // Ensure that processing and copying of the last cycle has finished
        cudaEventSynchronize(cycleDone[next_stream]);
        // Process current frame
        incKernel<<<grid, block, 0, stream[current_stream]>>>(d_data_out[current_stream],d_data_in[current_stream],N,inner_reps);
        // Upload next frame
        cudaMemcpyAsync(d_data_in[next_stream],h_data_in[next_stream],memsize,cudaMemcpyHostToDevice,stream[next_stream]);
        // Download current frame
        cudaMemcpyAsync(h_data_out[current_stream],d_data_out[current_stream],memsize,cudaMemcpyDeviceToHost,stream[current_stream]);

        cudaEventRecord(cycleDone[current_stream],stream[current_stream]);
        current_stream = next_stream;
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);

    return time;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Host Memory Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void init()
{
    for (int i=0; i<N; ++i)
    {
        h_data_source[i] = 0;
    }

    for (int i =0; i<STREAM_COUNT; ++i)
    {
        memcpy(h_data_in[i], h_data_source, memsize);
    }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%% End of FIR Filter Implementation %%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

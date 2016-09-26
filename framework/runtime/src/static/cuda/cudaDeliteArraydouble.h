#ifndef __cudaDeliteArraydouble__
#define __cudaDeliteArraydouble__

#include "DeliteCuda.h"

class cudaDeliteArraydouble {
public:
    double *data;
    int length;
    int offset;
    int stride;
    int flag;

    // Constructors
    __host__ __device__ cudaDeliteArraydouble(void) {
      length = 0;
      data = NULL;
    }

    __host__ cudaDeliteArraydouble(int _length) {
        length = _length;
        offset = 0;
        stride = 1;
        flag = 1;
        DeliteCudaMalloc((void**)&data,length*sizeof(double));
    }

    __host__ __device__ cudaDeliteArraydouble(int _length, double *_data, int _offset) {
        length = _length;
        data = _data;
        offset = _offset *_length;
        stride = 1;
        flag = 1;
    }

    __host__ __device__ cudaDeliteArraydouble(int _length, double *_data, int _offset, int _stride) {
        length = _length;
        data = _data;
        offset = _offset;
        stride = _stride;
        flag = 1;
    }

    __host__ __device__ double apply(int idx) {
      if(flag!=1)
        return data[offset + (idx % flag) * stride + idx / flag];
      else
        return data[offset + idx * stride];
    }

    __host__ __device__ void update(int idx, double value) {
      if(flag!=1)
        data[offset + (idx % flag) * stride + idx / flag] = value;
      else
        data[offset + idx * stride] = value;
    }

    // DeliteCoolection
    __host__ __device__ int size() {
        return length;
    }

    __host__ __device__ double dc_apply(int idx) {
        return apply(idx);
    }

    __host__ __device__ void dc_update(int idx, double value) {
        update(idx,value);
    }

    __host__ __device__ void dc_copy(cudaDeliteArraydouble from) {
      for(int i=0; i<length; i++)
        update(i,from.apply(i));
    }

    __host__ cudaDeliteArraydouble *dc_alloc(void) {
      return new cudaDeliteArraydouble(length);
    }

    __host__ cudaDeliteArraydouble *dc_alloc(int size) {
      return new cudaDeliteArraydouble(size);
    }
};

#endif
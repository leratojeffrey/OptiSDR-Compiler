#ifndef __cudaDeliteArrayfloat__
#define __cudaDeliteArrayfloat__

#include "DeliteCuda.h"

class cudaDeliteArrayfloat {
public:
    float *data;
    int length;
    int offset;
    int stride;
    int flag;

    // Constructors
    __host__ __device__ cudaDeliteArrayfloat(void) {
      length = 0;
      data = NULL;
    }

    __host__ cudaDeliteArrayfloat(int _length) {
        length = _length;
        offset = 0;
        stride = 1;
        flag = 1;
        DeliteCudaMalloc((void**)&data,length*sizeof(float));
    }

    __host__ __device__ cudaDeliteArrayfloat(int _length, float *_data, int _offset) {
        length = _length;
        data = _data;
        offset = _offset *_length;
        stride = 1;
        flag = 1;
    }

    __host__ __device__ cudaDeliteArrayfloat(int _length, float *_data, int _offset, int _stride) {
        length = _length;
        data = _data;
        offset = _offset;
        stride = _stride;
        flag = 1;
    }

    __host__ __device__ float apply(int idx) {
      if(flag!=1)
        return data[offset + (idx % flag) * stride + idx / flag];
      else
        return data[offset + idx * stride];
    }

    __host__ __device__ void update(int idx, float value) {
      if(flag!=1)
        data[offset + (idx % flag) * stride + idx / flag] = value;
      else
        data[offset + idx * stride] = value;
    }

    // DeliteCoolection
    __host__ __device__ int size() {
        return length;
    }

    __host__ __device__ float dc_apply(int idx) {
        return apply(idx);
    }

    __host__ __device__ void dc_update(int idx, float value) {
        update(idx,value);
    }

    __host__ __device__ void dc_copy(cudaDeliteArrayfloat from) {
      for(int i=0; i<length; i++)
        update(i,from.apply(i));
    }

    __host__ cudaDeliteArrayfloat *dc_alloc(void) {
      return new cudaDeliteArrayfloat(length);
    }

    __host__ cudaDeliteArrayfloat *dc_alloc(int size) {
      return new cudaDeliteArrayfloat(size);
    }
};

#endif
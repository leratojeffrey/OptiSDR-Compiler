#ifndef __cudaDeliteArrayint32_t__
#define __cudaDeliteArrayint32_t__

#include "DeliteCuda.h"

class cudaDeliteArrayint32_t {
public:
    int32_t *data;
    int length;
    int offset;
    int stride;
    int flag;

    // Constructors
    __host__ __device__ cudaDeliteArrayint32_t(void) {
      length = 0;
      data = NULL;
    }

    __host__ cudaDeliteArrayint32_t(int _length) {
        length = _length;
        offset = 0;
        stride = 1;
        flag = 1;
        DeliteCudaMalloc((void**)&data,length*sizeof(int32_t));
    }

    __host__ __device__ cudaDeliteArrayint32_t(int _length, int32_t *_data, int _offset) {
        length = _length;
        data = _data;
        offset = _offset *_length;
        stride = 1;
        flag = 1;
    }

    __host__ __device__ cudaDeliteArrayint32_t(int _length, int32_t *_data, int _offset, int _stride) {
        length = _length;
        data = _data;
        offset = _offset;
        stride = _stride;
        flag = 1;
    }

    __host__ __device__ int32_t apply(int idx) {
      if(flag!=1)
        return data[offset + (idx % flag) * stride + idx / flag];
      else
        return data[offset + idx * stride];
    }

    __host__ __device__ void update(int idx, int32_t value) {
      if(flag!=1)
        data[offset + (idx % flag) * stride + idx / flag] = value;
      else
        data[offset + idx * stride] = value;
    }

    // DeliteCoolection
    __host__ __device__ int size() {
        return length;
    }

    __host__ __device__ int32_t dc_apply(int idx) {
        return apply(idx);
    }

    __host__ __device__ void dc_update(int idx, int32_t value) {
        update(idx,value);
    }

    __host__ __device__ void dc_copy(cudaDeliteArrayint32_t from) {
      for(int i=0; i<length; i++)
        update(i,from.apply(i));
    }

    __host__ cudaDeliteArrayint32_t *dc_alloc(void) {
      return new cudaDeliteArrayint32_t(length);
    }

    __host__ cudaDeliteArrayint32_t *dc_alloc(int size) {
      return new cudaDeliteArrayint32_t(size);
    }
};

#endif
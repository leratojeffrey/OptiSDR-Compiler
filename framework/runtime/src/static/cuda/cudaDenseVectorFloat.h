#ifndef __cudaDenseVectorFloat__
#define __cudaDenseVectorFloat__
#include "cudaDeliteArrayfloat.h"
#include "cppDenseVectorFloat.h"
class cudaDenseVectorFloat {
public:
	cudaDeliteArrayfloat _data;
	int32_t _length;
	bool _isRow;
	__host__ __device__ cudaDenseVectorFloat(void) { }
	__host__ __device__ cudaDenseVectorFloat(cudaDeliteArrayfloat __data,int32_t __length,bool __isRow) {
		_data = __data;
		_length = __length;
		_isRow = __isRow;
	}
	__device__ void dc_copy(cudaDenseVectorFloat from) {
		_data.dc_copy(from._data);
		_length = from._length;
		_isRow = from._isRow;
	}
	__host__ cudaDenseVectorFloat *dc_alloc() {
		return new cudaDenseVectorFloat(*_data.dc_alloc(),_length,_isRow);
	}
	__host__ __device__ float dc_apply(int idx) {
		return _data.apply(idx);
	}
	__host__ __device__ void dc_update(int idx,float newVal) {
		_data.update(idx,newVal);
	}
	__host__ __device__ int dc_size(void) {
		return _data.length;
	}
};
class HostcudaDenseVectorFloat {
public:
cppDenseVectorFloat *host;
cudaDenseVectorFloat *dev;
};
#endif
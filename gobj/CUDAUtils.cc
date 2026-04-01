#include "CUDAUtils.hh"

#ifdef __CUDA__

#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

void cumeminfo(size_t* free, size_t* total){
  cudaError_t ret = cudaMemGetInfo(free, total);
  if(ret == cudaSuccess)
    cout << "cumeminfo - " << (*free/1.e9) << " of "
	 << (*total)/1.e9 << " GB free" << endl;
  else{
    cout << "cumeminfo - " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
  }
}

void cumalloc(void** ptr, size_t nbytes, string fn, string var){
  cudaError_t ret = cudaMalloc(ptr, nbytes);
  if(ret != cudaSuccess){
    cout << fn << " - " << cudaGetErrorString(ret) << ":" << endl
	 << "    failed to allocate "
	 << nbytes << " bytes for " << var << endl;
    size_t free, total;
    cumeminfo(&free, &total);
    cudaDeviceReset();
  }
}

void cumemcpy(void* dest, void* source, size_t nbytes,
	      bool to_dev, string fn, string var){
  cudaError_t ret = cudaMemcpy(dest, source, nbytes, to_dev ?
			       cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost);
  if(ret != cudaSuccess){
    cout << fn << " - " << cudaGetErrorString(ret) << ":" << endl
	 << "failed to copy " << nbytes << " bytes for " << var
	 << (to_dev?" to":" from") << " device" << endl;
    size_t free, total;
    cumeminfo(&free, &total);
    cudaDeviceReset();
  }
}

void cumemset(void* ptr, int value, size_t nbytes, string fn, string var){
  cudaError_t ret = cudaMemset(ptr, value, nbytes);
  if(ret != cudaSuccess){
    cout << fn << " - " << cudaGetErrorString(ret) << ":" << endl
	 << "failed to set " << nbytes << " bytes for " << var << endl;
    size_t free, total;
    cumeminfo(&free, &total);
    cudaDeviceReset();
  }
}

#endif

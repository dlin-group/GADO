#include "config_defs.hh"

#ifdef __CUDA__

#ifndef __CUDAUtils__
#define __CUDAUtils__

#include <string>

using namespace std;

void cumeminfo(size_t* free, size_t* total);

void cumalloc(void** ptr, size_t nbytes, string fn, string var);

void cumemcpy(void* dest, void* source, size_t nbytes,
	      bool to_dev, string fn, string var);

void cumemset(void* ptr, int val, size_t nbytes, string fn, string var);

#endif

#endif

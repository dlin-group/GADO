#ifndef __HDF5UTIL__
#define __HDF5UTIL__

#include <H5Cpp.h>
#include <cstring>
#include <string>
#include <vector>
#include <assert.h>
#include <iostream>
#include <assert.h>
#include <limits>
#include <utility>
#include <numeric>
#include <functional>

using namespace std;
using namespace H5;

template <typename T>
PredType GetHDF5PredType(){
  PredType dtype = PredType::NATIVE_B32;
  if(is_same<T, uint8_t>::value)       dtype = PredType::NATIVE_UINT8;
  else if(is_same<T, uint16_t>::value) dtype = PredType::NATIVE_UINT16;
  else if(is_same<T, uint32_t>::value) dtype = PredType::NATIVE_UINT32;
  else if(is_same<T, uint64_t>::value) dtype = PredType::NATIVE_UINT64;
  else if(is_same<T, int8_t>::value)   dtype = PredType::NATIVE_INT8;
  else if(is_same<T, int16_t>::value)  dtype = PredType::NATIVE_INT16;
  else if(is_same<T, int32_t>::value)  dtype = PredType::NATIVE_INT32;
  else if(is_same<T, int64_t>::value)  dtype = PredType::NATIVE_INT64;
  else if(is_same<T, float>::value)    dtype = PredType::NATIVE_FLOAT;
  else if(is_same<T, double>::value)   dtype = PredType::NATIVE_DOUBLE;
  else if(is_same<T, bool>::value)     dtype = PredType::NATIVE_HBOOL;
  else cout << "GetHDF5PredType: error, unsupported class type" << endl;
  return dtype;
}


template <typename T>
void WriteHDF5scalar(H5File* hfile, const T& v, string name){
  DataSpace fspace(H5S_SCALAR);
  PredType dtype = GetHDF5PredType<T>();
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace);
  dset.write(&v, dtype);
  fspace.close();
  dset.close();
}


template <typename T>
void WriteHDF5vector(H5File* hfile, const vector<T>& v, string name,
		     size_t chunk=0, int compression=0, size_t max_size=0){
  hsize_t dim[]    = {v.size()};
  hsize_t maxdim[] =
    {max_size == 0 ? v.size() : (max_size == numeric_limits<size_t>::max() ?
				 H5S_UNLIMITED : max_size)};
  hsize_t chs[]    = {(chunk == 0) ? v.size() : std::min(v.size(), chunk)};
  DataSpace fspace(1, dim, maxdim);
  PredType dtype = GetHDF5PredType<T>();
  DSetCreatPropList plist;
  const T val(0);
  plist.setFillValue(dtype, &val);
  plist.setChunk(1, chs);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace, plist);
  dset.write(&v.front(), dtype);
  fspace.close();
  plist.close();
  dset.close();
}


template <typename T>
void ExtendHDF5vector(H5File* hfile, const vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0};
  fspace.getSimpleExtentDims(dim);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t fstart[] = {dim[0]};
  hsize_t mstart[] = {0};
  hsize_t count[]  = {v.size()};
  dim[0] += v.size();
  dset.extend(dim);
  fspace = dset.getSpace();
  fspace.selectHyperslab(H5S_SELECT_SET, count, fstart);
  DataSpace mspace(1, count);
  mspace.selectHyperslab(H5S_SELECT_SET, count, mstart);
  dset.write(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  mspace.close();
  dset.close();
}


template <typename T>
void WriteHDF5matrix(H5File* hfile, const vector<T>& v, size_t nrows,
		     string name, size_t rchunk=0, size_t cchunk=0,
		     int compression=0, size_t max_rows=0, size_t max_cols=0){
  assert(v.size() % nrows == 0);
  size_t ncols = v.size() / nrows;
  hsize_t dim[] = {nrows, ncols};
  hsize_t maxdim[] =
    {max_rows == 0 ? nrows : (max_rows == numeric_limits<size_t>::max() ?
			      H5S_UNLIMITED : max_rows),
     max_cols==0 ? ncols : (max_cols == numeric_limits<size_t>::max() ?
			    H5S_UNLIMITED : max_cols)};
  hsize_t chs[] = {rchunk == 0 ? nrows : std::min(nrows, rchunk),
		   cchunk == 0 ? ncols : std::min(ncols, cchunk)};
  DataSpace fspace(2, dim, maxdim);
  PredType dtype = GetHDF5PredType<T>();
  DSetCreatPropList plist;
  const T val(0);
  plist.setFillValue(dtype, &val);
  plist.setChunk(2, chs);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace, plist);
  dset.write(&v.front(), dtype);
  fspace.close();
  plist.close();
  dset.close();
}


template <typename T>
void ExtendHDF5matrixRows(H5File* hfile, const vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim);
  assert(v.size() % dim[1] == 0);
  size_t nrows = v.size() / dim[1];
  PredType dtype = GetHDF5PredType<T>();
  hsize_t fstart[] = {dim[0], 0};
  hsize_t fcount[] = {nrows,  dim[1]};
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  dim[0] += nrows;
  dset.extend(dim);
  fspace = dset.getSpace();
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.write(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  fspace.close();
  mspace.close();
  dset.close();
}


template <typename T>
void ExtendHDF5matrixCols(H5File* hfile, const vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim);
  assert(v.size() % dim[1] == 0);
  size_t ncols = v.size() / dim[0];
  PredType dtype = GetHDF5PredType<T>();
  hsize_t fstart[] = {0, dim[1]};
  hsize_t fcount[] = {dim[0], ncols};
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  dim[1] += ncols;
  dset.extend(dim);
  fspace = dset.getSpace();
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.write(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  fspace.close();
  mspace.close();
  dset.close();
}


template <typename T>
void WriteHDF5arrayN(H5File* const hfile, vector<T>& v, vector<size_t> dims,
		     string name, vector<size_t> cchunk = vector<size_t>(0),
		     int compression=0,
		     vector<size_t> max_dims = vector<size_t>(0)){
  assert(dims.size() > 0 && dims.size() <= 32);
  if(cchunk.size()   != 0) assert(dims.size() == cchunk.size());
  if(max_dims.size() != 0) assert(dims.size() == max_dims.size());
  assert(accumulate(dims.begin(), dims.end(),
		    (size_t) 1, multiplies<size_t>()) == v.size());
  hsize_t dim[dims.size()], maxdim[dims.size()], chs[dims.size()];
  for(size_t n=0; n<dims.size(); n++){
    dim[n] = dims[n];
    maxdim[n] = (max_dims.size() == 0) ? dims[n] :
      (max_dims[n]==numeric_limits<size_t>::max() ? H5S_UNLIMITED:max_dims[n]);
    chs[n] = (cchunk.size() == 0) ? dims[n] : std::min(dims[n], cchunk[n]);
  }
  DataSpace fspace(dims.size(), dim, maxdim);
  PredType dtype = GetHDF5PredType<T>();
  DSetCreatPropList plist;
  const T val(0);
  plist.setFillValue(dtype, &val);
  plist.setChunk(dims.size(), chs);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace, plist);
  dset.write(&v.front(), dtype);
  fspace.close();
  plist.close();
  dset.close();
}


template <typename T>
void ExtendHDF5arrayN(H5File* hfile, const vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t dim[N];
  fspace.getSimpleExtentDims(dim);
  assert(v.size() % dim[0] == 0);
  size_t n = 1;
  for(int i=1; i<N; i++) n *= i;
  n = v.size() / n;
  PredType dtype = GetHDF5PredType<T>();
  hsize_t fstart[N], fcount[N];
  fstart[0] = dim[0];
  fcount[0] = n;
  for(int i=1; i<N; i++){
    fstart[i] = 0;
    fcount[i] = dim[i];
  }
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  dim[0] += n;
  dset.extend(dim);
  fspace = dset.getSpace();
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.write(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  fspace.close();
  mspace.close();
  dset.close();
}


template <typename T>
void ReadHDF5scalar(const H5File* hfile, T& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace(H5S_SCALAR);
  PredType dtype = GetHDF5PredType<T>();
  dset.read(&v, dtype);
  fspace.close();
  dset.close();
}


size_t HDF5vectorSize(const H5File* hfile, string name);

size_t HDF5vectorChunk(const H5File* hfile, string name);

size_t HDF5vectorNChunks(const H5File* hfile, string name);


template <typename T>
void ReadHDF5vector(const H5File* hfile, vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0};
  fspace.getSimpleExtentDims(dim, NULL);
  v.resize(dim[0]);
  PredType dtype = GetHDF5PredType<T>();
  dset.read(&v.front(), dtype);
  fspace.close();
  dset.close();
}


template <typename T>
void ReadHDF5vectorBlock(const H5File* hfile, vector<T>& v, string name,
			 size_t vchunk, size_t start=-1, size_t count=0){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[] = {0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(1, chunk);
  hsize_t fstart[] = {0};
  hsize_t fcount[] = {0};
  if(start < dim[0] && count != 0){
    fstart[0] = start;
    fcount[0] = std::min(count, (size_t) dim[0]-start);
  }
  else if(vchunk < (dim[0]/chunk[0] + (size_t) (dim[0]%chunk[0] != 0))){
    fstart[0] = vchunk * chunk[0];
    fcount[0] = std::min(chunk[0], dim[0]-fstart[0]);
  }
  else{
    v.resize(0);
    fspace.close();
    plist.close();
    dset.close();
  }
  v.resize(fcount[0]);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  plist.close();
  dset.close();
}


pair<size_t, size_t> HDF5matrixSize(const H5File* hfile, string name);

pair<size_t, size_t> HDF5matrixChunk(const H5File* hfile, string name);

pair<size_t, size_t> HDF5matrixNChunks(const H5File* hfile, string name);


template <typename T>
pair<size_t, size_t>  ReadHDF5matrix(const H5File* hfile,
				     vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  v.resize(dim[0] * dim[1]);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t fstart[] = {0, 0};
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, dim, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return make_pair(dim[0], dim[1]);
}


template <typename T>
pair<size_t, size_t> ReadHDF5matrixRows(const H5File* hfile, vector<T>& v,
					string name, size_t rchunk,
					size_t start=-1, size_t count=0){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[] = {0, 0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(2, chunk);
  hsize_t fstart[] = {0, 0};
  hsize_t fcount[] = {0, dim[1]};
  if(start < dim[0] && count != 0){
    fstart[0] = start;
    fcount[0] = std::min(count, (size_t) dim[0]-start);
  }
  else if(rchunk < (dim[0]/chunk[0] + (size_t) (dim[0]%chunk[0] != 0))){
    fstart[0] = rchunk * chunk[0];
    fcount[0] = std::min(chunk[0], dim[0]-fstart[0]);
  }
  else{
    v.resize(0);
    fspace.close();
    plist.close();
    dset.close();
    return make_pair(0, 0);
  }
  v.resize(fcount[0] * fcount[1]);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  plist.close();
  dset.close();
  return make_pair(fcount[0], fcount[1]);
}


template <typename T>
pair<size_t, size_t> ReadHDF5matrixCols(const H5File* hfile, vector<T>& v,
					string name, size_t cchunk,
					size_t start=-1, size_t count=0){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[] = {0, 0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(2, chunk);
  hsize_t fstart[] = {0, 0};
  hsize_t fcount[] = {dim[0], 0};
  if(start < dim[1] && count != 0){
    fstart[1] = start;
    fcount[1] = std::min(count, (size_t) dim[1]-start);
  }
  else if(cchunk < (dim[1]/chunk[1] + (size_t) (dim[1]%chunk[1] != 0))){
    fstart[1] = cchunk * chunk[1];
    fcount[1] = std::min(chunk[1], dim[1]-fstart[1]);
  }
  else{
    v.resize(0);
    fspace.close();
    plist.close();
    dset.close();
    return make_pair(0, 0);
  }
  v.resize(fcount[0] * fcount[1]);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  plist.close();
  dset.close();
  return make_pair(fcount[0], fcount[1]);
}


template <typename T>
pair<size_t, size_t> ReadHDF5matrixBlock(const H5File* hfile,
					 vector<T>& v, string name,
					 pair<size_t, size_t> cchunk,
					 pair<size_t, size_t> start,
					 pair<size_t, size_t> count){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[] = {0, 0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(2, chunk);
  hsize_t fstart[] = {0, 0};
  hsize_t fcount[] = {0, 0};
  if(start.first  < dim[0] && count.first  != 0 &&
     start.second < dim[1] && count.second != 0){
    fstart[0] = start.first;
    fstart[1] = start.second;
    fcount[0] = std::min(count.first,  (size_t) (dim[0]-fstart[0]));
    fcount[1] = std::min(count.second, (size_t) (dim[1]-fstart[1]));
  }
  else if(cchunk.first  < (dim[0]/chunk[0] + (size_t) (dim[0]%chunk[0] != 0)) &&
	  cchunk.second < (dim[1]/chunk[1] + (size_t) (dim[1]%chunk[1] != 0))){
    fstart[0] = cchunk.first  * chunk[0];
    fstart[1] = cchunk.second * chunk[1];
    fcount[0] = std::min(chunk[0], dim[0]-fstart[0]);
    fcount[1] = std::min(chunk[1], dim[1]-fstart[1]);
  }
  else{
    v.resize(0);
    fspace.close();
    dset.close();
    return make_pair(0, 0);
  }
  v.resize(fcount[0] * fcount[1]);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return make_pair(fcount[0], fcount[1]);
}


vector<size_t> HDF5arrayNSize(const H5File* hfile, string name);

vector<size_t> HDF5arrayNChunk(const H5File* hfile, string name);

vector<size_t> HDF5arrayNNChunks(const H5File* hfile, string name);


template <typename T>
vector<size_t> ReadHDF5arrayN(const H5File* hfile, vector<T>& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t dim[N];
  fspace.getSimpleExtentDims(dim, NULL);
  vector<size_t> dims(N);
  for(int i=0; i<N; i++) dims[i] = dim[i];
  v.resize(accumulate(dims.begin(), dims.end(),
		      (size_t) 1, multiplies<size_t>()));
  PredType dtype = GetHDF5PredType<T>();
  hsize_t fstart[N];
  for(int i=0; i<N; i++) fstart[i] = 0;
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, dim, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return dims;
}


template <typename T>
vector<size_t> ReadHDF5arrayNBlock(const H5File* hfile, vector<T>& v,
				   string name, vector<size_t> cchunk,
				   vector<size_t> start, vector<size_t> count){
  assert(cchunk.size() == start.size() && cchunk.size() == count.size());
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  assert(cchunk.size() == (size_t) N);
  hsize_t dim[N], chunk[N], fstart[N], fcount[N];
  fspace.getSimpleExtentDims(dim, NULL);
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(N, chunk);
  bool read_chunk = true, read_count = true;
  for(int i=0; i<N; i++){
    if(start[i] >= dim[i] || count[i] == 0) read_chunk = false;
    if(cchunk[i] >= (dim[i]/chunk[i] +
		     (size_t) (dim[i]%chunk[i] != 0))) read_count = false;
    if(!read_chunk && !read_count){
      v.resize(0);
      fspace.close();
      dset.close();
      return vector<size_t>(N, 0);
    }
  }
  size_t n = 1;
  vector<size_t> dims(N);
  for(int i=0; i<N; i++){
    fstart[i] = read_count ? start[i] : cchunk[i] * chunk[i];
    fcount[i] = read_count ?
      std::min(count[i], (size_t) (dim[i]-fstart[i])) :
      std::min(chunk[i], dim[i]-fstart[i]);
    n *= fcount[i];
    dims[i] = fcount[i];
  }
  v.resize(n);
  PredType dtype = GetHDF5PredType<T>();
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  dset.read(&v.front(), dtype, mspace, fspace);
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return dims;
}


template <>
void WriteHDF5scalar<string>(H5File* hfile, const string& v, string name);

template <>
void WriteHDF5vector<string>(H5File* hfile, const vector<string>& v,
			     string name, size_t, int, size_t max_size);

template <>
void ExtendHDF5vector<string>(H5File* hfile,
			      const vector<string>& v, string name);

template <>
void WriteHDF5matrix<string>(H5File* hfile, const vector<string>& v,
			     size_t nrows, string name,
			     size_t rchunk, size_t cchunk, int compression,
			     size_t max_rows, size_t max_cols);

template <>
void ExtendHDF5matrixRows<string>(H5File* hfile,
				  const vector<string>& v, string name);

template <>
void ExtendHDF5matrixCols<string>(H5File* hfile,
				  const vector<string>& v, string name);

template <>
void WriteHDF5arrayN<string>(H5File* const hfile, vector<string>& v,
			     vector<size_t> dims,
			     string name, vector<size_t>,
			     int compression, vector<size_t> max_dims);

template <>
void ExtendHDF5arrayN<string>(H5File* hfile,
			      const vector<string>& v, string name);

template <>
void ReadHDF5scalar<string>(const H5File* hfile, string& v, string name);

template <>
void ReadHDF5vector<string>(const H5File* hfile,
			    vector<string>& v, string name);

template <>
void ReadHDF5vectorBlock<string>(const H5File* hfile,
				 vector<string>& v, string name,
				 size_t, size_t start, size_t count);

template <>
pair<size_t, size_t> ReadHDF5matrix<string>(const H5File* hfile,
					    vector<string>& v,
					    string name);

template<> pair<size_t, size_t>
ReadHDF5matrixRows<string>(const H5File* hfile,
			   vector<string>& v, string name,
			   size_t, size_t start, size_t count);

template<> pair<size_t, size_t>
ReadHDF5matrixCols<string>(const H5File* hfile,
			   vector<string>& v, string name,
			   size_t, size_t start, size_t count);

template <>
pair<size_t, size_t> ReadHDF5matrixBlock<string>(const H5File* hfile,
						 vector<string>& v, string name,
						 pair<size_t, size_t>,
						 pair<size_t, size_t> start,
						 pair<size_t, size_t> count);

template <> vector<size_t>
ReadHDF5arrayN<string>(const H5File* hfile, vector<string>& v, string name);

template <> vector<size_t>
ReadHDF5arrayNBlock<string>(const H5File* hfile, vector<string>& v,
			    string name, vector<size_t> cchunk,
			    vector<size_t> start, vector<size_t> count);

#endif

#include "HDF5Util.hh"

using namespace std;
using namespace H5;

template <>
void WriteHDF5scalar<string>(H5File* hfile, const string& v, string name){
  DataSpace fspace(H5S_SCALAR);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace);
  char* v_char = new char(v.size()+1);
  strcpy(v_char, v.c_str());
  dset.write(v_char, dtype);
  delete v_char;
  fspace.close();
  dset.close();  
}


template <>
void WriteHDF5vector<string>(H5File* hfile, const vector<string>& v,
			     string name, size_t, int, size_t max_size){
  hsize_t dim[] = {v.size()};
  hsize_t maxdim[] =
    {max_size == 0 ? v.size() : (max_size == numeric_limits<size_t>::max() ?
				 H5S_UNLIMITED : max_size)};
  DataSpace fspace(1, dim, maxdim);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace);
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;    
  fspace.close();
  dset.close();
}


template <>
void ExtendHDF5vector<string>(H5File* hfile,
			      const vector<string>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0};
  fspace.getSimpleExtentDims(dim);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t fstart[] = {dim[0]};
  hsize_t mstart[] = {0};
  hsize_t count[]  = {v.size()};
  dim[0] += v.size();
  dset.extend(dim);
  fspace = dset.getSpace();
  fspace.selectHyperslab(H5S_SELECT_SET, count, fstart);
  DataSpace mspace(1, count);
  mspace.selectHyperslab(H5S_SELECT_SET, count, mstart);
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype, mspace, fspace);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;    
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  mspace.close();
  dset.close();
}


template <>
void WriteHDF5matrix<string>(H5File* hfile, const vector<string>& v,
			     size_t nrows, string name, size_t, size_t,
			     int, size_t max_rows, size_t max_cols){
  assert(v.size() % nrows == 0);
  size_t ncols = v.size() / nrows;
  hsize_t dim[] = {nrows, ncols};
  hsize_t maxdim[] =
    {max_rows == 0 ? nrows : (max_rows == numeric_limits<size_t>::max() ?
			      H5S_UNLIMITED : max_rows),
     max_cols==0 ? ncols : (max_cols == numeric_limits<size_t>::max() ?
			    H5S_UNLIMITED : max_cols)};
  DataSpace fspace(2, dim, maxdim);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace);
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  fspace.close();
  dset.close();
}


template <>
void ExtendHDF5matrixRows<string>(H5File* hfile,
				  const vector<string>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim);
  assert(v.size() % dim[1] == 0);
  size_t nrows = v.size() / dim[1];
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
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
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype, mspace, fspace);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  fspace.close();
  mspace.close();
  dset.close();
}


template <>
void ExtendHDF5matrixCols<string>(H5File* hfile,
				  const vector<string>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim);
  assert(v.size() % dim[1] == 0);
  size_t ncols = v.size() / dim[0];
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
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
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype, mspace, fspace);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  fspace.close();
  mspace.close();
  dset.close();
}


template <>
void WriteHDF5arrayN<string>(H5File* const hfile, vector<string>& v,
			     vector<size_t> dims,
			     string name, vector<size_t>,
			     int compression, vector<size_t> max_dims){
  assert(dims.size() > 0 && dims.size() <= 32);
  if(max_dims.size() != 0) assert(dims.size() == max_dims.size());
  assert(accumulate(dims.begin(), dims.end(),
		    (size_t) 1, multiplies<size_t>()) == v.size());
  hsize_t dim[dims.size()], maxdim[dims.size()];
  for(size_t n=0; n<dims.size(); n++){
    dim[n] = dims[n];
    maxdim[n] = (max_dims.size() == 0) ? dims[n] :
      (max_dims[n]==numeric_limits<size_t>::max() ? H5S_UNLIMITED:max_dims[n]);
  }
  DataSpace fspace(dims.size(), dim, maxdim);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  DataSet dset = hfile->createDataSet(name.c_str(), dtype, fspace);
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  fspace.close();
  dset.close();
}


template <>
void ExtendHDF5arrayN<string>(H5File* hfile,
			      const vector<string>& v, string name){
  DataSet dset = hfile->openDataSet(name);
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t dim[N];
  fspace.getSimpleExtentDims(dim);
  assert(v.size() % dim[0] == 0);
  size_t n = 1;
  for(int i=1; i<N; i++) n *= i;
  n = v.size() / n;
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
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
  char** v_char = new char*[v.size()];
  for(size_t i=0; i<v.size(); i++){
    v_char[i] = new char[v[i].size()+1];
    strcpy(v_char[i], v[i].c_str());
  }
  dset.write(v_char, dtype, mspace, fspace);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  fspace.close();
  mspace.close();
  dset.close();
}


size_t HDF5vectorSize(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0};
  fspace.getSimpleExtentDims(dim, NULL);
  fspace.close();
  dset.close();
  return dim[0];
}


size_t HDF5vectorChunk(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t chunk[] = {0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(1, chunk);
  fspace.close();
  plist.close();
  dset.close();
  return chunk[0];
}


size_t HDF5vectorNChunks(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[] = {0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(1, chunk);
  fspace.close();
  plist.close();
  dset.close();
  return dim[0]/chunk[0] + (size_t) (dim[0]%chunk[0] != 0); 
}


template <>
void ReadHDF5scalar<string>(const H5File* hfile, string& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace(H5S_SCALAR);
  char* v_char = NULL;
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  dset.read(v_char, dtype);
  v = string(v_char);
  if(v_char) delete v_char;
  fspace.close();
  dset.close();
}


template <>
void ReadHDF5vector<string>(const H5File* hfile,
			    vector<string>& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {1};
  fspace.getSimpleExtentDims(dim, NULL);
  v.resize(dim[0]);
  char** v_char = new char*[v.size()];
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  fspace.close();
  dset.close();
}


template <>
void ReadHDF5vectorBlock<string>(const H5File* hfile,
				 vector<string>& v, string name,
				 size_t, size_t start, size_t count){
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
  else{
    v.resize(0);
    fspace.close();
    plist.close();
    dset.close();
  }
  v.resize(fcount[0]);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  plist.close();
  dset.close();
}


pair<size_t, size_t> HDF5matrixSize(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  fspace.close();
  dset.close();
  return make_pair(dim[0], dim[1]);
}


pair<size_t, size_t> HDF5matrixChunk(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t chunk[] = {0, 0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(2, chunk);
  fspace.close();
  plist.close();
  dset.close();
  return make_pair(chunk[0], chunk[1]);
}


pair<size_t, size_t> HDF5matrixNChunks(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[] = {0, 0};
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(2, chunk);
  fspace.close();
  plist.close();
  dset.close();
  return make_pair(dim[0]/chunk[0] + (size_t) (dim[0]%chunk[0] != 0),
		   dim[1]/chunk[1] + (size_t) (dim[1]%chunk[1] != 0));
}


template <>
pair<size_t, size_t> ReadHDF5matrix<string>(const H5File* hfile,
					    vector<string>& v,
					    string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  v.resize(dim[0] * dim[1]);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t fstart[] = {0, 0};
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, dim, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return make_pair(dim[0], dim[1]);
}


template<> pair<size_t, size_t>
ReadHDF5matrixRows<string>(const H5File* hfile,
			   vector<string>& v, string name,
			   size_t, size_t start, size_t count){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t fstart[] = {0, 0};
  hsize_t fcount[] = {0, dim[1]};
  if(start < dim[0] && count != 0){
    fstart[0] = start;
    fcount[0] = std::min(count, (size_t) dim[0]-start);
  }
  else{
    v.resize(0);
    fspace.close();
    dset.close();
    return make_pair(0, 0);
  }
  v.resize(fcount[0] * fcount[1]);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return make_pair(fcount[0], fcount[1]);
}


template<> pair<size_t, size_t>
ReadHDF5matrixCols<string>(const H5File* hfile,
			   vector<string>& v, string name,
			   size_t, size_t start, size_t count){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t fstart[] = {0, 0};
  hsize_t fcount[] = {dim[0], 0};
  if(start < dim[1] && count != 0){
    fstart[1] = start;
    fcount[1] = std::min(count, (size_t) dim[1]-start);
  }
  else{
    v.resize(0);
    fspace.close();
    dset.close();
    return make_pair(0, 0);
  }
  v.resize(fcount[0] * fcount[1]);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return make_pair(fcount[0], fcount[1]);
}


template <>
pair<size_t, size_t> ReadHDF5matrixBlock<string>(const H5File* hfile,
						 vector<string>& v, string name,
						 pair<size_t, size_t>,
						 pair<size_t, size_t> start,
						 pair<size_t, size_t> count){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  hsize_t dim[] = {0, 0};
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t fstart[] = {start.first, start.second};
  if(fstart[0] >= dim[0] || fstart[1] >= dim[1]){
    v.resize(0);
    fspace.close();
    dset.close();
    return make_pair(0, 0);
  }
  hsize_t fcount[] = {std::min(count.first,  (size_t) (dim[0]-fstart[0])),
		      std::min(count.second, (size_t) (dim[1]-fstart[1]))};
  v.resize(fcount[0] * fcount[1]);
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return make_pair(fcount[0], fcount[1]);
}


template <> vector<size_t>
ReadHDF5arrayN<string>(const H5File* hfile, vector<string>& v, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t dim[N];
  fspace.getSimpleExtentDims(dim, NULL);
  vector<size_t> dims(N);
  for(int i=0; i<N; i++) dims[i] = dim[i];
  v.resize(accumulate(dims.begin(), dims.end(),
		      (size_t) 1, multiplies<size_t>()));
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t fstart[N];
  for(int i=0; i<N; i++) fstart[i] = 0;
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, dim, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return dims;
}


vector<size_t> HDF5arrayNSize(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t dim[N];
  fspace.getSimpleExtentDims(dim, NULL);
  fspace.close();
  dset.close();
  vector<size_t> dims(N);
  for(int i=0; i<N; i++) dims[i] = dim[i];
  return dims;
}


vector<size_t> HDF5arrayNChunk(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t chunk[N];
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(N, chunk);
  fspace.close();
  plist.close();
  dset.close();
  vector<size_t> ch(N);
  for(int i=0; i<N; i++) ch[i] = chunk[i];
  return ch;
}


vector<size_t> HDF5arrayNNChunks(const H5File* hfile, string name){
  DataSet dset = hfile->openDataSet(name.c_str());
  DataSpace fspace = dset.getSpace();
  int N = fspace.getSimpleExtentNdims();
  hsize_t dim[N];
  fspace.getSimpleExtentDims(dim, NULL);
  hsize_t chunk[N];
  DSetCreatPropList plist = dset.getCreatePlist();
  plist.getChunk(N, chunk);
  fspace.close();
  plist.close();
  dset.close();
  vector<size_t> chs(N);
  for(int i=0; i<N; i++)
    chs[i] = dim[i]/chunk[i] + (size_t) (dim[i]%chunk[i] != 0);
  return chs;
}


template <> vector<size_t>
ReadHDF5arrayNBlock<string>(const H5File* hfile, vector<string>& v,
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
  StrType dtype(PredType::C_S1, H5T_VARIABLE);
  hsize_t mstart[] = {0};
  hsize_t mcount[] = {v.size()};
  fspace.selectHyperslab(H5S_SELECT_SET, fcount, fstart);
  DataSpace mspace(1, mcount);
  mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart);
  char** v_char = new char*[v.size()];
  dset.read(v_char, dtype);
  for(size_t i=0; i<v.size(); i++) v[i] = string(v_char[i]);
  for(size_t i=0; i<v.size(); i++) delete [] v_char[i];
  delete [] v_char;
  mspace.selectNone();
  fspace.selectNone();
  mspace.close();
  fspace.close();
  dset.close();
  return dims;
}

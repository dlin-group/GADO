#ifndef __PMATRIX__
#define __PMATRIX__

#include "HDF5Util.hh"
#include <stdlib.h>
#include <string>
#include <cstring>
#include <regex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <set>
#include <random>
#include <thread>
#include <filesystem>
#include <TFile.h>
#include <TMatrixT.h>
#include <TRInterface.h>
#include <TRDataFrame.h>
#include <TRFunctionImport.h>
#include <H5Cpp.h>
#include <cblas.h>
#include <lapack.h>
#include <lapacke.h>

using namespace std;

template <typename T> class PMatrix{
public:
  PMatrix();
  PMatrix(string name, size_t nr, size_t nc, bool identity=false);
  PMatrix(string name, PMatrix<T> const& obj);
  PMatrix(string name, H5::H5File const& infile);
  PMatrix(string name, TFile* const infile);
  PMatrix(string name, string infile, ROOT::R::TRInterface& R);
  PMatrix(string name, string csv_fname,
	  vector<size_t> colincl = vector<size_t>(0),
	  vector<size_t> colskip = vector<size_t>(0));
  virtual ~PMatrix();

  bool operator==(PMatrix<T> const& obj);
  bool operator&(PMatrix<T> const& obj);
  bool operator|(PMatrix<T> const& obj);
  bool operator^(PMatrix<T> const& obj);
  
  PMatrix<T>* operator+(PMatrix<T> const& obj);
  PMatrix<T>* operator-(PMatrix<T> const& obj);
  PMatrix<T>* operator*(PMatrix<T>& obj);
  PMatrix<T>* operator*(T val);

  PMatrix<T>& operator+=(PMatrix<T> const& obj);
  PMatrix<T>& operator-=(PMatrix<T> const& obj);
  PMatrix<T>& operator*=(T val);

  virtual PMatrix<T>* axpy_oop(string name,T alpha, PMatrix<T> const& obj);
  virtual void axpy(T alpha, PMatrix<T> const& obj);
  virtual void RightMult(PMatrix<T>& obj, PMatrix<T>* out, bool zero=true);
  
  virtual string GetName()   const { return mname; }
  virtual size_t GetNRows()  const { return nrows; }
  virtual size_t GetNCols()  const { return ncols; }
  virtual size_t GetNData()  const { return ndata; }
  virtual bool GetUseCblas() const { return use_cblas; };
  virtual size_t GetNThreads() const;
  virtual T GetVal(size_t index) const;
  virtual T GetVal(size_t row, size_t col) const;
  virtual const vector<T>& GetDataConst() const { return data; }
  virtual vector<T>& GetData(){ return data; };
  virtual void SetData(vector<T>& d);
  virtual void SwapData(vector<T>& buffer);
  
  virtual void SetUseCblas(bool cblas) { use_cblas = cblas; }
  virtual void SetNThreads(size_t n);
  virtual void SetVal(size_t index, T val);
  virtual void SetVal(size_t row, size_t col, T val);
  virtual void AddVal(size_t index, T val);
  virtual void AddVal(size_t row, size_t col, T val);
  virtual void MultVal(size_t index, T val);
  virtual void MultVal(size_t row, size_t col, T val);

  virtual void SetAll(T val);
  virtual void Zero();
  virtual void AddIdentity(T scale=1);
  virtual void Transpose();
  virtual T Sum() const;
  virtual void Permute(vector<size_t>& permutation);
  virtual void PermuteRows(vector<size_t>& permutation);
  virtual void PermuteCols(vector<size_t>& permutation);
  virtual PMatrix<T>* Inverse(double* logdet=NULL);
  virtual void Inverse(PMatrix<T>* out, double* logdet=NULL);
  virtual void Invert(double* logdet=NULL);
  virtual PMatrix<T>* BlockwiseInverse(size_t minbl=16, double* logdet=NULL);
  virtual void BlockwiseInverse(PMatrix<T>* out,
				size_t minbl=16, double* logdet=NULL);
  virtual void BlockwiseInvert(size_t minbl=16, double* logdet=NULL);
  virtual PMatrix<T>* Sqrt();
  virtual void Sqrt(PMatrix<T>* out);
  virtual T Trace();
  virtual T TraceProduct(PMatrix<T>& obj);
  virtual T TraceProductSymm(PMatrix<T>& obj);
  virtual T TraceProductTranspose(PMatrix<T>& obj);
  
  virtual T InnerProduct(size_t i, size_t j);
  virtual T InnerProduct(size_t i, size_t j, PMatrix<T>& obj);
  virtual T InnerRowProduct(size_t i, size_t j);
  virtual T InnerRowProduct(size_t i, size_t j, PMatrix<T> const& obj);
  virtual T InnerColProduct(size_t i, size_t j);
  virtual T InnerColProduct(size_t i, size_t j, PMatrix<T>& obj);

  virtual PMatrix<T>* OuterProduct(size_t i, size_t j, T mult=T(1));
  virtual PMatrix<T>* OuterProduct(size_t i, size_t j,
				   PMatrix<T> const& obj, T mult=T(1));
  virtual void OuterProduct(size_t i, size_t j, T mult, PMatrix<T>* out);
  virtual void OuterProduct(size_t i, size_t j,
			    PMatrix<T> const& obj, T mult, PMatrix<T>* out);
  virtual PMatrix<T>* OuterRowProduct(size_t i, size_t j, T mult=T(1));
  virtual PMatrix<T>* OuterRowProduct(size_t i,size_t j,
				      PMatrix<T> const& obj, T mult=T(1));
  virtual void OuterRowProduct(size_t i, size_t j, T mult, PMatrix<T>* out);
  virtual void OuterRowProduct(size_t i,size_t j,
			       PMatrix<T> const& obj, T mult, PMatrix<T>* out);
  virtual PMatrix<T>* OuterColProduct(size_t i, size_t j, T mult=T(1));
  virtual PMatrix<T>* OuterColProduct(size_t, size_t j,
				      PMatrix<T>& obj, T mult=T(1));
  virtual void OuterColProduct(size_t i, size_t j, T mult, PMatrix<T>* out);
  virtual void OuterColProduct(size_t, size_t j,
			       PMatrix<T>& obj, T mult, PMatrix<T>*out);

  virtual PMatrix<T>* RowProjection(size_t i, size_t j);
  virtual PMatrix<T>* RowProjection(size_t i, size_t j, PMatrix<T>& obj);
  virtual void RowProjection(size_t i, size_t j, PMatrix<T>* out);
  virtual void RowProjection(size_t i, size_t j,PMatrix<T>& obj,PMatrix<T>* out);
  
  virtual void ScaleRow(size_t i, T val);
  virtual void ScaleCol(size_t i, T val);
  virtual void AddMultRow(size_t i, size_t j, T val);
  virtual void AddMultRow(size_t i, size_t j, T val, PMatrix<T>& obj);
  virtual void AddMultRowBuffer(size_t i, T val);
  virtual void AddMultCol(size_t i, size_t j, T val);
  virtual void AddMultCol(size_t i, size_t j, T val, PMatrix<T>& obj);
  virtual void AddMultColBuffer(size_t i, T val);

  virtual void SetRows(size_t i, size_t n, T val);
  virtual void SetRows(size_t i, size_t j, size_t n, PMatrix<T> const& obj);
  virtual void SetCols(size_t i, size_t n, T val);
  virtual void SetCols(size_t i, size_t j, size_t n, PMatrix<T> const& obj);
  virtual void InsertRows(size_t i, size_t j, size_t n, PMatrix<T> const& obj);
  virtual void InsertCols(size_t i, size_t j, size_t n, PMatrix<T> const& obj);
  virtual void DeleteRows(size_t i, size_t n);
  virtual void DeleteCols(size_t i, size_t n);

  virtual PMatrix<T>* GetBlock(size_t i, size_t m, size_t j, size_t n);
  virtual void GetBlock(size_t i, size_t j, PMatrix<T>* out);
  virtual void SetBlock(size_t i, size_t j, PMatrix<T> const& obj);
  virtual void ApplyThreshold(T thresh=T(0));
  virtual void ApplyLimits(T minval, T maxval);
  virtual void LimitL2Norm(T norm);
  virtual vector<set<size_t> > BlockDiagonalize(T thresh=T(0));

  virtual pair<T, T> RenormalizeRow(size_t i, T mean=T(0), T sigma=T(1));
  virtual pair<T, T> RenormalizeCol(size_t i, T mean=T(0), T sigma=T(1));
  
  virtual void Random();
  virtual void RandomSymmetric();
  virtual T GetRandom();

  virtual void RandomNormal();
  virtual void RandomNormalSymmetric();
  virtual T GetRandomNormal();

  virtual void RandomOrthonormal(size_t fixed_row=-1);

  virtual size_t BinValues(size_t max_bins);

  virtual bool IsFinite();
  virtual bool IsSymmetric(T thresh);
  virtual void MakeSymmetric(bool from_upper);

  virtual void Print(size_t max_size, size_t prec, string lead="") const;
  virtual void Print() const;

  virtual void WriteHDF5(H5::H5File &outfile, string name="",
			 size_t chunk_row=0, size_t chunk_col=0,
			 int compression=0) const;
  virtual void WriteROOT(TFile* outfile,      string name="") const;
  virtual void WriteRdata(string outfile,     string name,
			  ROOT::R::TRInterface& R);
  virtual vector<string> AddToREnv(string name, ROOT::R::TRInterface& R);

  virtual bool RowToBuffer(size_t i);
  virtual bool ColToBuffer(size_t i);
  virtual void ResetBuffer(bool zero=true);

protected:  
  string mname;
  size_t ndata;
  size_t nrows;
  size_t ncols;
  vector<T> data;
  vector<T> buff;
  size_t brow;
  size_t bcol;
  bool use_cblas;

  static random_device rdevice;
  static default_random_engine rengine;
  static uniform_real_distribution<double> uniform_real;
  static normal_distribution<double> normal_real;
};


template <typename T> random_device
PMatrix<T>::rdevice = { random_device() };
template <typename T> default_random_engine
PMatrix<T>::rengine = default_random_engine(rdevice());
template <typename T> uniform_real_distribution<double>
PMatrix<T>::uniform_real = uniform_real_distribution(0.0, 1.0);
template <typename T> normal_distribution<double>
PMatrix<T>::normal_real = normal_distribution(0.0, 1.0);

template <typename T> PMatrix<T>::PMatrix(){
  mname = "";
  ndata = 0;
  nrows = 0;
  ncols = 0;
  brow = -1;
  bcol = -1;
  use_cblas = true;
}

template <typename T>
PMatrix<T>::PMatrix(string name, size_t nr, size_t nc, bool identity) :
  PMatrix(){
  assert(name != "" && nr > 0 && nc > 0);
  mname = name;
  nrows = nr;
  ncols = nc;
  ndata = nrows * ncols;
  data = vector<T>(ndata);
  ResetBuffer();
  if(identity) this->AddIdentity();
}

template <typename T>
PMatrix<T>::PMatrix(string name, PMatrix<T> const& obj) :
  PMatrix(name, obj.GetNRows(), obj.GetNCols(), false){
  data = obj.data;
  use_cblas = obj.use_cblas;
}

template <typename T>
PMatrix<T>::PMatrix(string name, H5::H5File const& infile) :
  PMatrix(){
  assert(name != "" && infile.getFileSize() > 0);
  mname = name;
  H5::DataSet   dset   = infile.openDataSet(mname);
  H5::DataSpace dspace = dset.getSpace();
  H5T_class_t typeclass = dset.getTypeClass();
  size_t dsize = 0;
  size_t rank = dspace.getSimpleExtentNdims();
  assert(rank == 1 || rank == 2);
  hsize_t dims[2] = {1, 1};
  dspace.getSimpleExtentDims(dims, NULL);
  assert(dims[0] > 0 && dims[1] > 0);
  if(typeclass == H5T_INTEGER){
    H5::IntType inttype = dset.getIntType();
    dsize = inttype.getSize();
    if(is_same<T, int>::value){
      if(sizeof(int) != dsize){
	cout << "PMatrix of type int, but HDF5 int size is " << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(dims[0] * dims[1]);
	dset.read(&data.front(), H5::PredType::NATIVE_INT);
      }
    }
    else if(is_same<T, long>::value){
      if(sizeof(long) != dsize){
	cout << "PMatrix of type long, but HDF5 int size is " << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(dims[0] * dims[1]);
	dset.read(&data.front(), H5::PredType::NATIVE_LONG);
      }
    }
    else{
      cout << "PMatrix unsupported int type of size " << dsize << endl;
      dspace.close(); dset.close();
      return;
    }
  }
  else if(typeclass == H5T_FLOAT){
    H5::FloatType floattype = dset.getFloatType();
    dsize = floattype.getSize();
    if(is_same<T, float>::value){
      if(sizeof(float) != dsize){
	cout << "PMatrix of type float, but HDF5 float size is "
	     << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(dims[0] * dims[1]);
	dset.read(&data.front(), H5::PredType::NATIVE_FLOAT);
      }
    }
    else if(is_same<T, double>::value){
      if(sizeof(double) != dsize){
	cout << "PMatrix of type double, but HDF5 float size is "
	     << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(dims[0] * dims[1]);
	dset.read(&data.front(), H5::PredType::NATIVE_DOUBLE);
      }
    }
    else{
      cout << "PMatrix unsupported float type of size " << dsize << endl;
      dspace.close(); dset.close();
      return;
    }
  }
  else{
    cout << "PMatrix unsupported HDF5 type class " << typeclass << endl;
    dspace.close(); dset.close();
    return;
  }
  dspace.close(); dset.close();
  mname = name;
  nrows = dims[0];
  ncols = dims[1];
  ndata = nrows * ncols;
  ResetBuffer();
}

template <typename T> PMatrix<T>::PMatrix(string name, TFile* const infile) :
  PMatrix<T>(){
  mname = name;
  TMatrixT<T>* m = (TMatrixT<T>*) infile->Get(name.c_str());
  nrows = m->GetNrows();
  ncols = m->GetNcols();
  ndata = nrows * ncols;
  data  = vector<T>(m->GetMatrixArray(), m->GetMatrixArray() + ndata);
  ResetBuffer();
}

template <typename T> PMatrix<T>::PMatrix(string name, string infile,
					  ROOT::R::TRInterface& R) :
  PMatrix<T>(){
  mname = name;
  ROOT::R::TRDataFrame df;
  R << ("load(file.path(\"" + infile + "\"), dfenv <- new.env())").c_str();
  R << ("df <- dfenv$" + name).c_str();
  R["df"] >> df;
  ROOT::R::TRFunctionImport asMatrix("as.matrix");
  TMatrixT<T> m = Rcpp::as<TMatrixT<T> >(asMatrix(df));
  nrows = m.GetNrows();
  ncols = m.GetNcols();
  ndata = nrows * ncols;
  data = vector<T>(m.GetMatrixArray(), m.GetMatrixArray() + ndata);
  ResetBuffer();
}

template <typename T> PMatrix<T>::PMatrix(string name, string csv_name,
					  vector<size_t> colincl,
					  vector<size_t> colskip) :
  PMatrix<T>(){
  ifstream csv_file(csv_name);
  if(!csv_file.is_open()){
    cout << "PMatrix::PMatrix(name, csv_name) error - cannot open "
	 << csv_name << endl;
    return;
  }
  if(!csv_file.good()){
    cout << "PMatrix::PMatrix(name, csv_name) eror - cannot read "
	 << " headers from " << csv_name << endl;
    return;
  }
  string line, hname;
  vector<string> header;
  getline(csv_file, line);
  line = regex_replace(line, regex("\r"), "");
  stringstream ss_head(line);
  size_t index = -1;
  while(getline(ss_head, hname, ',')){
    index ++;
    if(colincl.size() > 0){
      if(find(colincl.begin(), colincl.end(), index) == colincl.end())
	continue;
    }
    else if(colskip.size() > 0){
      if(find(colskip.begin(), colskip.end(), index) != colskip.end())
	continue;
    }
    header.push_back(hname);
  }
  T val;
  size_t tmp_size;
  size_t iline = 0;
  while(getline(csv_file, line)){
    iline ++;
    line = regex_replace(line, regex("\r"), "");
    stringstream ss(line);
    tmp_size = data.size();
    index = -1;
    while(ss.peek() == '"'){
      ss.ignore();
      while(ss.peek() != '"') ss.ignore();
      ss.ignore();
      if(ss.peek() == ',') ss.ignore();
      index ++;
    }
    while(ss >> val){
      index ++;
      if(ss.peek() == '"'){
	ss.ignore();
	while(ss.peek() != '"') ss.ignore();
	ss.ignore();
      }
      if(ss.peek() == ',') ss.ignore();
      if(colincl.size() > 0){
	if(find(colincl.begin(), colincl.end(), index) == colincl.end())
	  continue;
      }
      else if(colskip.size() > 0){
	if(find(colskip.begin(), colskip.end(), index) != colskip.end())
	  continue;
      }
      data.push_back(val);
    }
    if(data.size() - tmp_size != header.size()){
      cout << "PMatrix::PMatrix(name, csv_name) error - expected row "
	   << "of size " << header.size() << ", but read size "
	   << data.size() - tmp_size << endl;
      csv_file.close();
      return;
    }
  }
  csv_file.close();
  nrows = data.size()/header.size();
  ncols = header.size();
  ndata = nrows * ncols;
  assert(name != "" && nrows > 0 && ncols > 0);
  mname = name;
  ResetBuffer();
}

template <typename T> PMatrix<T>::~PMatrix(){
}

template <typename T> bool PMatrix<T>::operator==(PMatrix<T> const& obj){
  if(!(*this & obj)) return false;
  for(size_t i=0; i<ndata; i++) if(data[i] != obj.data[i]) return false;
  return true;
}

template <typename T> bool PMatrix<T>::operator&(PMatrix<T> const& obj){
  return (nrows == obj.GetNRows() && ncols == obj.GetNCols());
}

template <typename T> bool PMatrix<T>::operator|(PMatrix<T> const& obj){
  return ncols == obj.GetNRows();
}

template <typename T> bool PMatrix<T>::operator^(PMatrix<T> const& obj){
  return nrows == obj.GetNCols();
}

template <typename T> PMatrix<T>* PMatrix<T>::operator+(PMatrix<T> const& obj){
  if(*this & obj) return axpy_oop(mname+"_"+obj.GetName()+"_sum", T(1), obj);
  else return new PMatrix<T>();
}

template <typename T> PMatrix<T>* PMatrix<T>::operator-(PMatrix<T> const& obj){
  if(*this & obj) return axpy_oop(mname+"_"+obj.GetName()+"_diff", T(-1), obj);
  else return new PMatrix<T>();
}

template <typename T> PMatrix<T>* PMatrix<T>::operator*(PMatrix<T>& obj){
  if(*this | obj){
    PMatrix<T>* out = new PMatrix<T>(mname+"_"+obj.GetName()+"_mult",
				     nrows, obj.ncols, false);
    out->use_cblas = use_cblas;
    RightMult(obj, out);
    return out;
  }
  else return new PMatrix<T>();
}

template <typename T> PMatrix<T>* PMatrix<T>::operator*(T val){
  PMatrix<T>* out = new PMatrix<T>(mname+"_scal", *this);
  (*out) *= val;
  return out;
}

template <typename T> PMatrix<T>& PMatrix<T>::operator+=(PMatrix<T>const& obj){
  axpy(T(1), obj);
  return *this;
}

template <typename T> PMatrix<T>& PMatrix<T>::operator-=(PMatrix<T>const& obj){
  axpy(T(-1), obj);
  return *this;
}

template <typename T> PMatrix<T>& PMatrix<T>::operator*=(T val){
  if(val == T(1)) return *this;
  if(use_cblas && is_same<T, float>::value)
    cblas_sscal(ndata, (float) val, (float*) &data.front(), 1);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dscal(ndata, (double) val, (double*) &data.front(), 1);
  else
    for(size_t i=0; i<ndata; i++) data[i] *= val;
  brow = -1; bcol = -1;
  return *this;
}

template <typename T> PMatrix<T>* PMatrix<T>::axpy_oop(string name, T alpha,
						       PMatrix<T> const& obj){
  if(!(*this & obj)) return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix(name, *this);
  if(use_cblas && is_same<T, float>::value && ndata <= INT_MAX)
    cblas_saxpy(out->ndata, (float) alpha,
		(float*) &obj.data.front(), 1, (float*) &out->data.front(), 1);
  else if(use_cblas && is_same<T, double>::value && ndata <= INT_MAX)
    cblas_daxpy(out->ndata, (double) alpha,
		(double*)&obj.data.front(), 1, (double*)&out->data.front(), 1);
  else if(alpha == T(1))
    for(size_t i=0; i<out->ndata; i++) out->data[i] += obj.data[i];
  else if(alpha == T(-1))
    for(size_t i=0; i<out->ndata; i++) out->data[i] -= obj.data[i];
  else
    for(size_t i=0; i<out->ndata; i++) out->data[i] += alpha*obj.data[i];
  brow = -1; bcol = -1;
  return out;
}

template <typename T> void PMatrix<T>::axpy(T alpha, PMatrix<T> const& obj){
  if(!(*this & obj)) return;
  if(use_cblas && is_same<T, float>::value && ndata <= INT_MAX)
    cblas_saxpy(ndata, (float) alpha,
		(float*) &obj.data.front(), 1, (float*) &data.front(), 1);
  else if(use_cblas && is_same<T, float>::value && ndata <= INT_MAX)
    cblas_daxpy(ndata, (double) alpha,
		(double*) &obj.data.front(), 1, (double*) &data.front(), 1);
  else if(alpha == T(1))
    for(size_t i=0; i<ndata; i++) data[i] += obj.data[i];
  else if(alpha == T(-1))
    for(size_t i=0; i<ndata; i++) data[i] -= obj.data[i];
  else
    for(size_t i=0; i<ndata; i++) data[i] += alpha * obj.data[i];
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::RightMult(PMatrix<T>& obj,
						 PMatrix<T>* out, bool zero){
  if(out){
    if(zero) out->Zero();
    if(*this | obj){
      if(use_cblas && is_same<T, float>::value)
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    nrows, obj.ncols, ncols, (float) 1.0,
		    (float*) &data.front(),      ncols,
		    (float*) &obj.data.front(),  obj.ncols, (float) 1.0,
		    (float*) &out->data.front(), out->ncols);
      else if(use_cblas && is_same<T, double>::value)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    nrows, obj.ncols, ncols, (double) 1.0,
		    (double*) &data.front(),      ncols,
		    (double*) &obj.data.front(),  obj.ncols, (double) 1.0,
		    (double*) &out->data.front(), out->ncols);
      else{
	size_t offset = 0;
	for(size_t i=0; i<obj.ncols; i++){
	  offset = i;
	  for(size_t j=0; j<nrows; (j++, offset+=obj.ncols))
	    out->data[offset] = InnerProduct(j, i, obj);
	}
      }
    }
  }
  else out = (*this) * obj;
}

template<typename T> size_t PMatrix<T>::GetNThreads() const{
  if(use_cblas) return openblas_get_num_threads();
  else return 1;
}

template <typename T> T PMatrix<T>::GetVal(size_t index) const{
  return (index < ndata) ? data[index] : T();
}

template <typename T> T PMatrix<T>::GetVal(size_t row, size_t col) const{
  return (row < nrows && col < ncols) ? data[row*ncols+col] : T();
}

template <typename T> void PMatrix<T>::SetNThreads(size_t n){
  if(use_cblas){
    n = max((size_t) 1, min(n, (size_t) thread::hardware_concurrency()));
    if(GetNThreads() != n) openblas_set_num_threads(n);
  }
}

template <typename T> void PMatrix<T>::SetVal(size_t index, T val){
  if(index < ndata){
    data[index] = val;
    if(index/ncols == brow) brow = -1;
    if(index%ncols == bcol) bcol = -1;
  }
}

template <typename T> void PMatrix<T>::SetVal(size_t row, size_t col, T val){
  if(row < nrows && col < ncols){
    data[row*ncols+col] = val;
    if(row == brow) brow = -1;
    if(col == bcol) bcol = -1;
  }
}

template <typename T> void PMatrix<T>::AddVal(size_t index, T val){
  if(index < ndata){
    data[index] += val;
    if(index/ncols == brow) brow = -1;
    if(index%ncols == bcol) bcol = -1;
  }
}

template <typename T> void PMatrix<T>::AddVal(size_t row, size_t col, T val){
  if(row < nrows && col < ncols){
    data[row*ncols+col] += val;
    if(row == brow) brow = -1;
    if(col == bcol) bcol = -1;
  }
}

template <typename T> void PMatrix<T>::MultVal(size_t index, T val){
  if(index < ndata){
    data[index] *= val;
    if(index/ncols == brow) brow = -1;
    if(index%ncols == bcol) bcol = -1;
  }
}

template <typename T> void PMatrix<T>::MultVal(size_t row, size_t col, T val){
  if(row < nrows && col < ncols){
    data[row*ncols+col] *= val;
    if(row == brow) brow = -1;
    if(col == bcol) bcol = -1;
  }
}

template <typename T> void PMatrix<T>::SetData(vector<T>& d){
  assert(d.size() == ndata);
  copy(d.begin(), d.end(), data.begin());
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::SwapData(vector<T>& buffer){
  assert(buffer.size() >= ndata);
  size_t tmp = buffer.size();
  buffer.resize(ndata);
  data.resize(tmp);
  data.swap(buffer);
}

template <typename T> void PMatrix<T>::SetAll(T val){
  fill(data.begin(), data.end(), val);
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::Zero(){
  SetAll(T());
}

template <typename T> void PMatrix<T>::AddIdentity(T scale){
  size_t index = 0, count = 0;
  while(count < nrows && count < ncols){
    data[index] += scale;
    index += ncols + 1;
    count ++;
  }
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::Transpose(){
  if(nrows == 1){
    nrows = ncols;
    ncols = 1;
  }
  else if(ncols == 1){
    ncols = nrows;
    nrows = 1;
  }
  else if(nrows == ncols){
    T tmp;
    size_t offset = 0, cindex = 0;
    for(size_t rindex=0; rindex<ndata; rindex++){
      cindex = nrows + offset * (nrows + 1);
      rindex += offset;
      offset ++;
      for(; cindex<ndata; cindex+=nrows){
	rindex ++;
	tmp = data[rindex];
	data[rindex] = data[cindex];
	data[cindex] = tmp;
      }
    }
    cindex = nrows;
    nrows  = ncols;
    ncols  = cindex;
  }
  else{
    vector<T> tmp(data.begin(), data.end());
    size_t i, j;
    for(size_t r=0; r<nrows; r++){
      for(size_t c=0; c<ncols; c++){
	  i = r*ncols + c;
	  j = c*nrows + r;
	  data[j] = tmp[i];
      }
    }
    i = nrows;
    nrows  = ncols;
    ncols  = i;
  }
  size_t b = brow;
  brow = bcol;
  bcol = b;
}

template <typename T> T PMatrix<T>::Sum() const{
  return accumulate(data.begin(), data.end(), 0.0);
}

template <typename T> void PMatrix<T>::Permute(vector<size_t>& permutation){
  PermuteCols(permutation);
  PermuteRows(permutation);
}

template <typename T> void PMatrix<T>::PermuteRows(vector<size_t>& permutation){
  if(nrows == 1) return;
  Transpose();
  PermuteCols(permutation);
  Transpose();
}

template <typename T>
void PMatrix<T>::PermuteCols(vector<size_t>& permutation){
  if(ncols == 1) return;
  if(permutation.size() != ncols) return;
  fill(buff.begin(), buff.end(), T(0));
  brow = -1; bcol = -1;
  for(size_t i=0; i<nrows; i++){
    transform(permutation.begin(), permutation.end(), buff.begin(),
	      [this, &i](const size_t j){ return GetVal(i, j); });
    copy(buff.begin(), buff.begin()+ncols, data.begin()+i*ncols);
  }
}

template <typename T> PMatrix<T>* PMatrix<T>::Inverse(double* logdet){
  PMatrix<T>* inv = new PMatrix<T>(mname+"_inv", *this);
  Inverse(inv, logdet);
  return inv;
}

template <typename T> void PMatrix<T>::Inverse(PMatrix<T>* out,double* logdet){
  if(nrows != ncols || !out) return;
  if(nrows != out->GetNRows() || ncols != out->GetNCols()) return;
  out->SetData(GetData());
  if(nrows == 1){
    if(logdet) *logdet += log(data[0]);
    out->data[0] = T(1) / data[0];
    return;
  }
  else if(nrows == 2){
    double det = data[0]*data[3] - data[1]*data[2];
    if(logdet) *logdet += log(det);
    out->data[0] =  data[3] / det;
    out->data[1] = -data[1] / det;
    out->data[2] = -data[2] / det;
    out->data[3] =  data[0] / det;
    return;
  }
  vector<int> pivot(nrows);
  if(use_cblas && is_same<T, float>::value){
    LAPACKE_sgetrf(CblasRowMajor, nrows, ncols,
		   (float*) &out->data.front(), ncols, &pivot.front());
    if(logdet)
      for(size_t i=0; i<out->GetNRows(); i++)
	*logdet += log(abs(out->GetVal(i, i)));
    LAPACKE_sgetri(CblasRowMajor, nrows,
		   (float*) &out->data.front(), ncols, &pivot.front());
  }
  else if(use_cblas && is_same<T, double>::value){
    LAPACKE_dgetrf(CblasRowMajor, nrows, ncols,
		   (double*) &out->data.front(), ncols, &pivot.front());
    if(logdet)
      for(size_t i=0; i<out->GetNRows(); i++)
	*logdet += log(abs(out->GetVal(i, i)));
    LAPACKE_dgetri(CblasRowMajor, nrows,
		   (double*) &out->data.front(), ncols, &pivot.front());
  }
  else{
    PMatrix<T>* identity = new PMatrix<T>(mname+"_tmp", nrows, ncols, true);
    out->InsertCols(ncols, 0, ncols, *identity);
    delete identity;
    size_t joffset = 0;
    for(size_t i=0; i<nrows; i++){
      out->ScaleRow(i, out->data[i*(1+out->ncols)]);
      joffset = 0;
      for(size_t j=0; j<nrows; (j++, joffset+=out->ncols)){
	if(i == j) continue;
	out->AddMultRow(j, i, -out->data[i+joffset]);
      }
    }
    out->DeleteCols(0, nrows);
  }
}

template <typename T> void PMatrix<T>::Invert(double* logdet){
  if(nrows != ncols) return;
  if(nrows == 1){
    if(logdet) *logdet += log(data[0]);
    data[0] = T(1) / data[0];
    return;
  }
  else if(nrows == 2){
    double det = data[0]*data[3] - data[1]*data[2];
    if(logdet) *logdet += log(det);
    T tmp = data[0];
    data[0] =  data[3] / det;
    data[1] = -data[1] / det;
    data[2] = -data[2] / det;
    data[3] =      tmp / det;
    return;
  }
  vector<int> pivot(nrows);
  if(is_same<T, float>::value){
    LAPACKE_sgetrf(CblasRowMajor, nrows, ncols,
		   (float*) &data.front(), ncols, &pivot.front());
    if(logdet)
      for(size_t i=0; i<GetNRows(); i++)
	*logdet += log(abs(GetVal(i, i)));
    LAPACKE_sgetri(CblasRowMajor, nrows,
		   (float*) &data.front(), ncols, &pivot.front());
  }
  else if(is_same<T, double>::value){
    LAPACKE_dgetrf(CblasRowMajor, nrows, ncols,
		   (double*) &data.front(), ncols, &pivot.front());
    if(logdet)
      for(size_t i=0; i<GetNRows(); i++)
	*logdet += log(abs(GetVal(i, i)));
    LAPACKE_dgetri(CblasRowMajor, nrows,
		   (double*) &data.front(), ncols, &pivot.front());
  }
  else{
    PMatrix<T>* tmp = new PMatrix("tmp", *this);
    Inverse(tmp, logdet);
    SetData(tmp->GetData());
    delete tmp;
  }
}

template <typename T> PMatrix<T>* PMatrix<T>::BlockwiseInverse(size_t minbl,
							       double* logdet){
  PMatrix<T>* inv = NULL;
  BlockwiseInverse(inv, minbl, logdet);
  return inv;
}

template <typename T> void PMatrix<T>::BlockwiseInverse(PMatrix<T>* out,
							size_t minbl,
							double* logdet){
  if(nrows != ncols) return;
  if(out){
    if(nrows != out->GetNRows() || ncols != out->GetNCols()) return;
    out->Zero();
  }
  else{
    out = new PMatrix<T>(mname+"_inv", nrows, ncols, false);
    out->use_cblas = use_cblas;
  }
  minbl = 1 << (size_t) ceil(log2(minbl));
  if(nrows < 2*minbl || minbl < 2) return Inverse(out, logdet);
  const size_t blrows = 1 << ((size_t) ceil(log2(nrows)) - 1);
  PMatrix<T>* Ainv = GetBlock(0, blrows, 0, blrows);
  if(blrows == minbl) Ainv->Invert(logdet);
  else Ainv->BlockwiseInvert(minbl, logdet);
  PMatrix<T>* block = GetBlock(blrows, blrows, 0, blrows);
  PMatrix<T>* CAinv = (*block) * (*Ainv);
  GetBlock(0, blrows, block);
  PMatrix<T>* AinvB  = (*Ainv)  * (*block);
  PMatrix<T>* CAinvB = (*CAinv) * (*block);
  GetBlock(blrows, blrows, block);
  for(size_t i=nrows-blrows; i<blrows; i++) block->SetVal(i, i, T(1));
  (*block) -= (*CAinvB);
  if(blrows == minbl) block->Invert(logdet);
  else block->BlockwiseInvert(minbl, logdet);
  out->SetBlock(blrows, blrows, *block);
  PMatrix<T>* AinvBDinv = CAinvB;
  AinvB->RightMult(*block, AinvBDinv);
  (*AinvBDinv) *= -1.0;
  out->SetBlock(0, blrows, *AinvBDinv);
  PMatrix<T>* DinvCAinv = AinvBDinv;
  block->RightMult(*CAinv, DinvCAinv);
  delete block;
  (*DinvCAinv) *= -1.0;
  out->SetBlock(blrows, 0, *DinvCAinv);
  PMatrix<T>* AinvBDinvCAinv = CAinv;
  AinvB->RightMult(*DinvCAinv, AinvBDinvCAinv);
  delete AinvB;
  delete DinvCAinv;
  (*Ainv) -= (*AinvBDinvCAinv);
  delete AinvBDinvCAinv;
  out->SetBlock(0, 0, *Ainv);
  delete Ainv;
}

template <typename T> void PMatrix<T>::BlockwiseInvert(size_t minbl,
						       double* logdet){
  if(nrows != ncols) return;
  minbl = 1 << (size_t) ceil(log2(minbl));
  if(nrows < 2*minbl || minbl < 2) return Invert(logdet);
  const size_t blrows = 1 << ((size_t) ceil(log2(nrows)) - 1);
  PMatrix<T>* Ainv = GetBlock(0, blrows, 0, blrows);
  if(blrows == minbl) Ainv->Invert(logdet);
  else Ainv->BlockwiseInvert(minbl, logdet);
  PMatrix<T>* block = GetBlock(blrows, blrows, 0, blrows);
  PMatrix<T>* CAinv = (*block) * (*Ainv);
  GetBlock(0, blrows, block);
  PMatrix<T>* AinvB  = (*Ainv)  * (*block);
  PMatrix<T>* CAinvB = (*CAinv) * (*block);
  GetBlock(blrows, blrows, block);
  for(size_t i=nrows-blrows; i<blrows; i++) block->SetVal(i, i, T(1));
  (*block) -= (*CAinvB);
  if(blrows == minbl) block->Invert(logdet);
  else block->BlockwiseInvert(minbl, logdet);
  SetBlock(blrows, blrows, *block);
  PMatrix<T>* AinvBDinv = CAinvB;
  AinvB->RightMult(*block, AinvBDinv);
  (*AinvBDinv) *= -1.0;
  SetBlock(0, blrows, *AinvBDinv);
  PMatrix<T>* DinvCAinv = AinvBDinv;
  block->RightMult(*CAinv, DinvCAinv);
  delete block;
  (*DinvCAinv) *= -1.0;
  SetBlock(blrows, 0, *DinvCAinv);
  PMatrix<T>* AinvBDinvCAinv = CAinv;
  AinvB->RightMult(*DinvCAinv, AinvBDinvCAinv);
  delete AinvB;
  delete DinvCAinv;
  (*Ainv) -= (*AinvBDinvCAinv);
  delete AinvBDinvCAinv;
  SetBlock(0, 0, *Ainv);
  delete Ainv;
}

template <typename T> PMatrix<T>* PMatrix<T>::Sqrt(){
  if(nrows != ncols) return new PMatrix<T>();
  PMatrix<T>* out = NULL;
  Sqrt(out);
  return out;
}

template <typename T> void PMatrix<T>::Sqrt(PMatrix<T>* out){
  if(nrows != ncols) return;
  if(!out) out = new PMatrix<T>(mname+"_sqrt", nrows, ncols, false);
  else{
    out->Zero();
    if(out->GetNRows() != nrows || out->GetNCols() != ncols) return;
  }
  if(nrows == 1)
    out->data[0] = sqrt(data[0]);
  else if(nrows == 2){
    T s = sqrt(data[0]*data[3] - data[1]*data[2]);
    T t = sqrt(data[0]+data[3] + 2*s);
    out->data[0] = (data[0]+s) / t;
    out->data[1] = data[1] / t;
    out->data[2] = data[2] / t;
    out->data[3] = (data[3]+s) / t;
  }
  else if(is_same<T, double>::value){
    if(!IsSymmetric(T(1.0e-8)))
      cout << "PMatrix<T>::Sqrt() is only supported for symmetric matrices "
	   << "of size greater than 2" << endl;
    int il = 0, iu = 0, m = 0;
    double abstol = -1.0, vl = 0.0, vu = 0.0;
    vector<double> w(nrows), z(nrows*nrows), b(nrows*nrows);
    vector<int> isuppz(2*nrows);
    LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'A', 'U',
		   nrows, (double*) &data.front(), nrows, vl, vu, il, iu,
		   abstol, &m, &w.front(), &z.front(), nrows, &isuppz.front());
    size_t k = 0;
    for(size_t i=0; i<nrows; i++){
      double s = sqrt(w[i]);
      for(size_t j=0; j<nrows; j++){
	k = j*nrows+i;
	b[k] = z[k] * s;
      }
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		nrows, nrows, nrows, 1, &b.front(), nrows,
		&z.front(), nrows, 0, (double*) &out->data.front(), nrows);
  }
  else
    cout << "PMatrix<T>::Sqrt() is not implemented for this data type" << endl;
}

template <typename T> T PMatrix<T>::Trace(){
  size_t n = std::min(nrows, ncols);
  double tr = 0.0;
  for(size_t i=0; i<n; i++) tr += GetVal(i, i);
  return (T) tr;
}

template <typename T> T PMatrix<T>::TraceProduct(PMatrix<T>& obj){
  if(!((*this) | obj)) return T();
  size_t n = std::min(nrows, obj.ncols);
  double tr = 0.0;
  for(size_t i=0; i<n; i++) tr += InnerProduct(i, i, obj);
  return (T) tr;
}

template <typename T> T PMatrix<T>::TraceProductSymm(PMatrix<T>& obj){
  if(!(*this & obj)) return T();
  return inner_product(data.begin(), data.end(), obj.data.begin(), 0.0);
}

template <typename T> T PMatrix<T>::TraceProductTranspose(PMatrix<T>& obj){
  if(ncols != obj.ncols) return T();
  size_t n = std::min(nrows, obj.nrows);
  double tr = 0.0;
  for(size_t i=0; i<n; i++) tr += InnerRowProduct(i, i, obj);
  return (T) tr;
}

template <typename T> T PMatrix<T>::InnerProduct(size_t i, size_t j){
  if(nrows != ncols || i >= nrows || !ColToBuffer(j)) return T();
  auto row = data.begin() + i*ncols;
  if(use_cblas && is_same<T, float>::value)
    return cblas_sdot(ncols, (float*) &*row, 1, (float*) &buff.front(), 1);
  else if(use_cblas && is_same<T, double>::value)
    return cblas_ddot(ncols, (double*) &*row, 1, (double*) &buff.front(), 1);
  else{
    double sum = 0.0;
    for(size_t m=0; m<ncols; m++) sum += ((double)*(row+m))*((double)buff[m]);
    return (T) sum;
  }
}

template <typename T>
T PMatrix<T>::InnerProduct(size_t i, size_t j, PMatrix<T>& obj){
  if(!(*this|obj) || i >= nrows || !obj.ColToBuffer(j)) return T();
  auto row = data.begin() + i*ncols;
  if(use_cblas && is_same<T, float>::value)
    return cblas_sdot(ncols, (float*) &*row, 1, (float*) &obj.buff.front(), 1);
  else if(use_cblas && is_same<T, double>::value)
    return cblas_ddot(ncols, (double*)&*row, 1, (double*)&obj.buff.front(), 1);
  else{
    double sum = 0.0;
    for(size_t m=0; m<ncols; m++)sum+=((double)*(row+m))*((double)obj.buff[m]);
    return (T) sum;
  }
}

template <typename T> T PMatrix<T>::InnerRowProduct(size_t i, size_t j){
  if(i >= nrows || j >= nrows) return T();
  auto rowi = data.begin() + i*ncols;
  auto rowj = data.begin() + j*ncols;
  if(use_cblas && is_same<T, float>::value)
    return cblas_sdot(ncols, (float*) &*rowi, 1, (float*) &*rowj, 1);
  else if(use_cblas && is_same<T, double>::value)
    return cblas_ddot(ncols, (double*) &*rowi, 1, (double*) &*rowj, 1);
  else{
    double sum = 0.0;
    for(size_t m=0; m<ncols; m++)
      sum += ((double)*(rowi+m)) * ((double)*(rowj+m));
    return (T) sum;
  }
}

template <typename T>
T PMatrix<T>::InnerRowProduct(size_t i, size_t j, PMatrix<T> const& obj){
  if(ncols != obj.ncols || i >= nrows || j >= obj.nrows)return T();
  auto rowi = data.begin() + i*ncols;
  auto rowj = obj.data.begin() + j*obj.ncols;
  if(use_cblas && is_same<T, float>::value)
    return cblas_sdot(ncols, (float*) &*rowi, 1, (float*) &*rowj, 1);
  else if(use_cblas && is_same<T, double>::value)
    return cblas_ddot(ncols, (double*) &*rowi, 1, (double*) &*rowj, 1);
  else{
    double sum = 0.0;
    for(size_t m=0; m<ncols; m++)
      sum += ((double)*(rowi+m)) * ((double)*(rowj+m));
    return (T) sum;
  }
}

template <typename T> T PMatrix<T>::InnerColProduct(size_t i, size_t j){
  if(!ColToBuffer(i) || j >= ncols) return T();
  vector<T> b = vector<T>(nrows);
  for(size_t m=0; m<nrows; (m++, j+=ncols)) b[m] = data[j];
  if(use_cblas && is_same<T, float>::value)
    return cblas_sdot(nrows, (float*) &buff.front(), 1,
		      (float*) &b.front(), 1);
  else if(use_cblas && is_same<T, double>::value)
    return cblas_ddot(nrows, (double*) &buff.front(), 1,
		      (double*) &b.front(), 1);
  else{
    double sum = 0.0;
    for(size_t m=0; m<nrows; m++) sum += ((double)buff[m]) * ((double)b[m]);
    return (T) sum;
  }
}

template <typename T>
T PMatrix<T>::InnerColProduct(size_t i, size_t j, PMatrix<T>& obj){
  if(nrows != obj.nrows || !ColToBuffer(i) || !obj.ColToBuffer(j)) return T();
  if(use_cblas && is_same<T, float>::value)
    return cblas_sdot(nrows, (float*) &buff.front(), 1,
		      (float*) &obj.buff.front(), 1);
  else if(use_cblas && is_same<T, double>::value)
    return cblas_ddot(nrows, (double*) &buff.front(), 1,
		      (double*) &obj.buff.front(), 1);
  else{
    double sum = 0.0;
    for(size_t m=0; m<nrows; m++) sum+=((double)buff[m])*((double)obj.buff[m]);
    return (T) sum;
  }
}

template <typename T>
PMatrix<T>* PMatrix<T>::OuterProduct(size_t i, size_t j, T mult){
  if(nrows != ncols || !ColToBuffer(i) || j >= nrows) return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix<T>(mname+"_oprod_"+
				  to_string(i)+"_"+to_string(j),
				  nrows, ncols, false);
  out->use_cblas = use_cblas;
  OuterProduct(i, j, mult, out);
  return out;
}

template <typename T>
PMatrix<T>* PMatrix<T>::OuterProduct(size_t i, size_t j,
				     PMatrix<T> const& obj, T mult){
  if(!(*this^obj) || !ColToBuffer(i) || j >= obj.nrows)return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix<T>(mname+"_"+obj.mname+"_oprod_"+
				   to_string(i)+"_"+to_string(j),
				   nrows, obj.ncols, false);
  out->use_cblas = use_cblas;
  OuterProduct(i, j, obj, mult, out);
  return out;
}

template <typename T>
void PMatrix<T>::OuterProduct(size_t i, size_t j, T mult, PMatrix<T>* out){
  if(nrows != ncols || !ColToBuffer(i) || j >= nrows) return;
  if(!out) out = OuterProduct(i, j, mult);
  if(out->nrows != nrows || out->ncols != ncols) return;
  auto row = data.begin() + j*ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_sger(CblasRowMajor, nrows, ncols, (float) mult,
	       (float*) &buff.front(), 1, (float*) &*row, 1,
	       (float*) &out->data.front(), ncols);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dger(CblasRowMajor, nrows, ncols, (double) mult,
	       (double*) &buff.front(), 1, (double*) &*row, 1,
	       (double*) &out->data.front(), ncols);
  else{
    size_t moffset = 0;
    for(size_t m=0; m<nrows; (m++, moffset+=ncols))
      for(size_t n=0; n<ncols; n++)
	out->data[moffset+n] += buff[m] * (*(row+n)) * mult;
  }
}

template <typename T>
void PMatrix<T>::OuterProduct(size_t i, size_t j,
			      PMatrix<T> const& obj, T mult, PMatrix<T>* out){
  if(!(*this^obj) || !ColToBuffer(i) || j >= obj.nrows) return;
  if(!out) out = OuterProduct(i, j, obj, mult);
  if(out->nrows != nrows || out->ncols != obj.ncols) return;
  auto row = obj.data.begin() + j*obj.ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_sger(CblasRowMajor, nrows, ncols, (float) mult,
	       (float*) &buff.front(), 1, (float*) &*row, 1,
	       (float*) &out->data.front(), ncols);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dger(CblasRowMajor, nrows, ncols, (double) mult,
	       (double*) &buff.front(), 1, (double*) &*row, 1,
	       (double*) &out->data.front(), ncols);
  else{
    size_t moffset = 0;
    for(size_t m=0; m<nrows; (m++, moffset+=ncols))
      for(size_t n=0; n<obj.ncols; n++)
	out->data[moffset+n] += buff[m] * (*(row+n)) * mult;
  }
}

template <typename T>
PMatrix<T>* PMatrix<T>::OuterRowProduct(size_t i, size_t j, T mult){
  if(i >= nrows || j >= nrows) return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix<T>(mname+"_orprod_"+
				   to_string(i)+"_"+to_string(j),
				   ncols, ncols, false);
  out->use_cblas = use_cblas;
  OuterRowProduct(i, j, mult, out);
  return out;
}

template <typename T>
PMatrix<T>* PMatrix<T>::OuterRowProduct(size_t i, size_t j,
					PMatrix<T> const& obj, T mult){
  if(ncols != obj.ncols || i >= nrows || j >=obj.nrows)return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix<T>(mname+"_"+obj.mname+"_prprod_"+
				   to_string(i)+"_"+to_string(j),
				   obj.ncols, ncols, false);
  out->use_cblas = use_cblas;
  OuterRowProduct(i, j, obj, mult, out);
  return out;
}

template <typename T>
void PMatrix<T>::OuterRowProduct(size_t i, size_t j, T mult, PMatrix<T>* out){
  if(i >= nrows || j >= nrows) return;
  if(!out) out = OuterRowProduct(i, j, mult);
  if(out->nrows != ncols || out->ncols != ncols) return;
  auto rowi = data.begin() + i*ncols;
  auto rowj = data.begin() + j*ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_sger(CblasColMajor, ncols, ncols, (float) mult,
	       (float*) &*rowj, 1, (float*) &*rowi, 1,
	       (float*) &out->data.front(), ncols);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dger(CblasColMajor, ncols, ncols, (double) mult,
	       (double*) &*rowj, 1, (double*) &*rowi, 1,
	       (double*) &out->data.front(), ncols);
  else{
    size_t k = 0;
    const auto si = rowi + ncols;
    const auto sj = rowj + ncols;
    auto rowk = rowj;
    for(; rowi!=si; rowi++)
      for(rowk=rowj; rowk!=sj; (rowk++, k++))
	out->data[k] += (*rowi) * (*rowk) * mult;
  }
}

template <typename T>
void PMatrix<T>::OuterRowProduct(size_t i, size_t j, PMatrix<T> const& obj,
				 T mult, PMatrix<T>* out){
  if(ncols != obj.ncols || i >= nrows || j >=obj.nrows) return;
  if(!out) out = OuterRowProduct(i, j, obj, mult);
  if(out->nrows != obj.ncols || out->ncols != ncols) return;
  auto rowi = data.begin() + i*ncols;
  auto rowj = obj.data.begin() + j*obj.ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_sger(CblasColMajor, ncols, obj.ncols, (float) mult,
	       (float*) &*rowj, 1, (float*) &*rowi, 1,
	       (float*) &out->data.front(), ncols);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dger(CblasColMajor, ncols, obj.ncols, (double) mult,
	       (double*) &*rowj, 1, (double*) &*rowi, 1,
	       (double*) &out->data.front(), ncols);
  else{
    size_t k = 0;
    const auto si = rowi + ncols;
    const auto sj = rowj + obj.ncols;
    auto rowk = rowj;
    for(; rowi!=si; rowi++)
      for(rowk=rowj; rowk!=sj; (rowk++, k++))
	out->data[k] += (*rowi) * (*rowk) * mult;
  }
}

template <typename T>
PMatrix<T>* PMatrix<T>::OuterColProduct(size_t i, size_t j, T mult){
  if(!ColToBuffer(i) || j >= ncols) return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix<T>(mname+"_ocprod_"+
				   to_string(i)+"_"+to_string(j),
				   nrows, nrows, false);
  out->use_cblas = use_cblas;
  OuterColProduct(i, j, mult, out);
  return out;
}

template <typename T>
PMatrix<T>* PMatrix<T>::OuterColProduct(size_t i, size_t j,
					PMatrix<T>& obj, T mult){
  if(nrows != obj.nrows || !ColToBuffer(i) || !obj.ColToBuffer(j))
    return new PMatrix<T>();
  PMatrix<T>* out = new PMatrix<T>(mname+"_"+obj.mname+"_ocprod_"+
				   to_string(i)+"_"+to_string(j),
				   obj.nrows, nrows, false);
  out->use_cblas = use_cblas;
  OuterColProduct(i, j, obj, mult, out);
  return out;
}

template <typename T>
void PMatrix<T>::OuterColProduct(size_t i, size_t j, T mult, PMatrix<T>* out){
  if(!ColToBuffer(i) || j >= ncols) return;
  if(!out) out = OuterColProduct(i, j, mult);
  if(out->nrows != nrows || out->ncols != nrows) return;
  vector<T> b = vector<T>(nrows);
  size_t moffset = j;
  for(size_t m=0; m<nrows; (m++, moffset+=ncols)) b[m] = data[moffset];
  if(use_cblas && is_same<T, float>::value)
    cblas_sger(CblasColMajor, nrows, nrows, (float) mult,
	       (float*) &b.front(), 1, (float*) &buff.front(), 1,
	       (float*) &out->data.front(), nrows);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dger(CblasColMajor, nrows, nrows, (double) mult,
	       (double*) &b.front(), 1, (double*) &buff.front(), 1,
	       (double*) &out->data.front(), nrows);
  else{
    moffset = 0;
    for(size_t m=0; m<nrows; (m++, moffset+=nrows))
      for(size_t n=0; n<nrows; n++)
	out->data[moffset+n] += b[n] * buff[m] * mult;
  }
}

// define == operator and have special case for *this==obj calling above
template <typename T>
void PMatrix<T>::OuterColProduct(size_t i, size_t j,
				 PMatrix<T>& obj, T mult, PMatrix<T>* out){
  if(nrows != obj.nrows || !ColToBuffer(i) || !obj.ColToBuffer(j)) return;
  if(!out) out = OuterColProduct(i, j, obj, mult);
  if(out->nrows != obj.nrows || out->ncols != nrows) return;
  if(use_cblas && is_same<T, float>::value)
    cblas_sger(CblasColMajor, obj.nrows, nrows, (float) mult,
	       (float*) &obj.buff.front(), 1, (float*) &buff.front(), 1,
	       (float*) &out->data.front(), nrows);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dger(CblasColMajor, obj.nrows, nrows, (double) mult,
	       (double*) &obj.buff.front(), 1, (double*) &buff.front(), 1,
	       (double*) &out->data.front(), nrows);
  else{
    size_t moffset = 0;
    for(size_t m=0; m<obj.nrows; (m++, moffset+=nrows))
      for(size_t n=0; n<nrows; n++)
	out->data[moffset+n] += obj.buff[n] * buff[m] * mult;
  }
}

template <typename T>
PMatrix<T>* PMatrix<T>::RowProjection(size_t i, size_t j){
  if(i >= nrows || j >= nrows) return NULL;
  PMatrix<T>* out = GetBlock(j, 1, 0, ncols);
  *out *= InnerRowProduct(i, j) / InnerRowProduct(j, j);
  return out;
}

template <typename T>
PMatrix<T>* PMatrix<T>::RowProjection(size_t i, size_t j, PMatrix<T>& obj){
  if(i >= nrows || j >= obj.GetNRows() || ncols != obj.GetNCols()) return NULL;
  PMatrix<T>* out = obj.GetBlock(j, 1, 0, obj.ncols);
  *out *= InnerRowProduct(i, j, obj) / obj.InnerRowProduct(j, j);
  return out;
}

template <typename T>
void PMatrix<T>::RowProjection(size_t i, size_t j, PMatrix<T>* out){
  if(out == NULL){
    out = RowProjection(i, j);
    return;
  }
  else if(i >= nrows || j >= ncols) return;
  else if(out->GetNCols() != ncols || out->GetNRows() != 1) return;
  GetBlock(j, 0, out);
  *out *= InnerRowProduct(i, j) / InnerRowProduct(j, j);
}

template <typename T>
  void PMatrix<T>::RowProjection(size_t i, size_t j,
			       PMatrix<T>& obj, PMatrix<T>* out){
  if(out == NULL){
    out = RowProjection(i, j, obj);
    return;
  }
  else if(i >= nrows || j >= obj.GetNRows() || ncols != obj.GetNCols()) return;
  else if(out->GetNCols() != ncols || out->GetNRows() != 1) return;
  obj.GetBlock(j, 0, out);
  *out *= InnerRowProduct(i, j, obj) / obj.InnerRowProduct(j, j);
}
  
template <typename T> void PMatrix<T>::ScaleRow(size_t i, T val){
  if(i >= nrows) return;
  auto row = data.begin() + i*ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_sscal(ncols, (float) (1.0/val), (float*) &*row, 1);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dscal(ncols, (double) (1.0/val), (double*) &*row, 1);
  else
    for(size_t m=0; m<ncols; m++) (*(row+m)) /= val;
}

template <typename T> void PMatrix<T>::ScaleCol(size_t i, T val){
  if(i >= ncols) return;
  if(use_cblas && (is_same<T, float>::value || is_same<T, double>::value)){
    ColToBuffer(i);
    if(is_same<T, float>::value)
      cblas_sscal(nrows, (float) (1.0/val), (float*) &buff.front(), 1);
    else
      cblas_dscal(nrows, (double) (1.0/val), (double*) &buff.front(), 1);
    for(size_t j=0; j<nrows; (j++, i+=ncols)) data[i] = buff[j];
  }
  else
    for(; i<ndata; i+= ncols) data[i] /= val;
}

template <typename T> void PMatrix<T>::AddMultRow(size_t i, size_t j, T val){
  if(i >= nrows || j >= nrows) return;
  auto rowi = data.begin() + i*ncols;
  auto rowj = data.begin() + j*ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_saxpy(ncols, (float) val, (float*) &*rowj, 1, (float*) &*rowi, 1);
  else if(use_cblas && is_same<T, double>::value)
    cblas_daxpy(ncols, (double) val, (double*) &*rowj, 1, (double*) &*rowi, 1);
  else{
    if(val == T(1))
      for(size_t m=0; m<ncols; m++) (*(rowi+m)) += (*(rowj+m));
    else
      for(size_t m=0; m<ncols; m++) (*(rowi+m)) += (*(rowj+m)) * val;
  }
}

template <typename T>
void PMatrix<T>::AddMultRow(size_t i, size_t j, T val, PMatrix<T>& obj){
  if(i >= nrows || j >= obj.nrows || ncols != obj.ncols) return;
  auto rowi = data.begin() + i*ncols;
  auto rowj = obj.data.begin() + j*obj.ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_saxpy(ncols, (float) val, (float*) &*rowj, 1, (float*) &*rowi, 1);
  else if(use_cblas && is_same<T, double>::value)
    cblas_daxpy(ncols, (double) val, (double*) &*rowj, 1, (double*) &*rowi, 1);
  else{
    if(val == T(1))
      for(size_t m=0; m<ncols; m++) (*(rowi+m)) += (*(rowj+m));
    else
      for(size_t m=0; m<ncols; m++) (*(rowi+m)) += (*(rowj+m)) * val;
  }
}

template <typename T> void PMatrix<T>::AddMultRowBuffer(size_t i, T val){
  if(i >= nrows) return;
  auto rowi = data.begin() + i*ncols;
  if(use_cblas && is_same<T, float>::value)
    cblas_saxpy(ncols, (float) val,
		(float*) &buff.front(), 1, (float*) &*rowi, 1);
  else if(use_cblas && is_same<T, double>::value)
    cblas_daxpy(ncols, (double) val,
		(double*) &buff.front(), 1, (double*) &*rowi, 1);
  else{
    if(val == T(1))
      for(size_t m=0; m<ncols; m++) (*(rowi+m)) += buff[m];
    else
      for(size_t m=0; m<ncols; m++) (*(rowi+m)) += buff[m] * val;
  }
}

template <typename T> void PMatrix<T>::AddMultCol(size_t i, size_t j, T val){
  if(i >= ncols || j >= ncols) return;
  if(use_cblas && (is_same<T, float>::value || is_same<T, double>::value)){
    ColToBuffer(i);
    vector<T> b = vector<T>(nrows);
    size_t moffset = j;
    for(size_t m=0; m<nrows; (m++, moffset+=ncols)) b[m] = data[moffset];
    if(is_same<T, float>::value)
      cblas_saxpy(nrows, (float) val,
		  (float*) &b.front(), 1, (float*) &buff.front(), 1);
    else
      cblas_daxpy(nrows, (double) val,
		  (double*) &b.front(), 1, (double*) &buff.front(), 1);
    for(size_t k=0; k<nrows; (k++, i+=ncols)) data[i] = buff[k];
  }
  else{
    if(val == T(1))
      for(; i<ndata; (i+=ncols, j+=ncols)) data[i] += data[j];
    else
      for(; i<ndata; (i+=ncols, j+=ncols)) data[i] += data[j] * val;
  }
}

template <typename T>
void PMatrix<T>::AddMultCol(size_t i, size_t j, T val, PMatrix<T>& obj){
  if(i >= ncols || j >= obj.ncols || nrows != obj.nrows) return;
  if(use_cblas && (is_same<T, float>::value || is_same<T, double>::value)){
    ColToBuffer(i);
    obj.ColToBuffer(j);
    if(is_same<T, float>::value)
      cblas_saxpy(nrows, (float) val,
		  (float*) &obj.buff.front(), 1, (float*) &buff.front(), 1);
    else
      cblas_daxpy(nrows, (double) val,
		  (double*) &obj.buff.front(), 1, (double*) &buff.front(), 1);
    for(size_t k=0; k<nrows; (k++, i+=ncols)) data[i] = buff[k];
  }
  else{
    if(val == T(1))
      for(; i<ndata; (i+=ncols, j+=ncols)) data[i] += obj.data[j];
    else
      for(; i<ndata; (i+=ncols, j+=ncols)) data[i] += obj.data[j] * val;
  }
}

template <typename T> void PMatrix<T>::AddMultColBuffer(size_t i, T val){
  if(i >= ncols) return;
  if(use_cblas && (is_same<T, float>::value || is_same<T, double>::value)){
    vector<T> b = vector<T>(nrows);
    size_t moffset = i;
    for(size_t m=0; m<nrows; (m++, moffset+=ncols)) b[m] = data[moffset];
    if(is_same<T, float>::value)
      cblas_saxpy(nrows, (float) val,
		  (float*) &buff.front(), 1, (float*) &b.front(), 1);
    else
      cblas_daxpy(nrows, (double) val,
		  (double*) &buff.front(), 1, (double*) &b.front(), 1);
    for(size_t k=0; k<nrows; (k++, i+=ncols)) data[i] = b[k];
  }
  else{
    if(val == T(1))
      for(size_t k=0; k<nrows; (k++, i+=ncols)) data[i] += buff[k];
    else
      for(size_t k=0; k<nrows; (k++, i+=ncols)) data[i] += buff[k] * val;
  }
}

template <typename T> void PMatrix<T>::SetRows(size_t i, size_t n, T val){
  if(i >= nrows || i+n > nrows) return;
  i *= ncols;
  n = i + n*ncols;
  for(; i<n; i++) data[i] = val; 
}

template <typename T>
void PMatrix<T>::SetRows(size_t i, size_t j, size_t n, PMatrix<T> const& obj){
  if(i >= nrows     || i+n > nrows     ||
     j >= obj.nrows || j+n > obj.nrows || ncols != obj.ncols) return;
  n = (j+n) * obj.ncols;
  i *= ncols;
  for(j*=obj.ncols; j<n; (j++, i++)) data[i] = obj.data[j];
}

template <typename T> void PMatrix<T>::SetCols(size_t i, size_t n, T val){
  if(i >= ncols || i+n > ncols) return;
  n += i;
  for(; i<ndata; (i+=ncols, n+=ncols))
    for(size_t m=i; m<n; m++) data[m] = val;
}

template <typename T>
void PMatrix<T>::SetCols(size_t i, size_t j, size_t n, PMatrix<T> const& obj){
  if(i >= ncols     || i+n > ncols     ||
     j >= obj.ncols || j+n > obj.ncols || nrows != obj.nrows) return;
  n += j;
  for(; j<obj.ndata; (j+=obj.ncols, n+=obj.ncols, i+=ncols-1))
    for(size_t m=j; m<n; (m++, i++)) data[i] = obj.data[m];
}

template <typename T>
void PMatrix<T>::InsertRows(size_t i,size_t j,size_t n, PMatrix<T> const& obj){
  if(i   >  nrows    || j     >= obj.nrows ||
     j+n > obj.nrows || ncols != obj.ncols);
  nrows += n;
  ndata = nrows * ncols;
  data.insert(data.begin()     + i*ncols,
	      obj.data.begin() + j*obj.ncols,
	      obj.data.begin() + j*obj.ncols + n*obj.ncols);
  ResetBuffer();
}

template <typename T>
void PMatrix<T>::InsertCols(size_t i,size_t j,size_t n, PMatrix<T> const& obj){
  if(i   > ncols      || j     >= obj.ncols ||
     j+n >  obj.ncols || nrows != obj.nrows) return;
  vector<T> tmpdata = data;
  size_t tmpncols = ncols;
  ncols += n;
  ndata = nrows * ncols;
  data.clear();
  for(size_t m=0; m<nrows; m++){
    data.insert(data.begin()     + m*ncols,
		tmpdata.begin()  + m*tmpncols,
		tmpdata.begin()  + m*tmpncols  + i);
    data.insert(data.begin()     + m*ncols     + i,
		obj.data.begin() + m*obj.ncols + j,
		obj.data.begin() + m*obj.ncols + j + n);
    data.insert(data.begin()     + m*ncols     + i + n,
		tmpdata.begin()  + m*tmpncols  + i,
		tmpdata.begin()  + m*tmpncols  + i + tmpncols - i);
  }
  ResetBuffer();
}

template <typename T> void PMatrix<T>::DeleteRows(size_t i, size_t n){
  if(i >= nrows || i+n > nrows) return;
  nrows -= n;
  ndata = nrows * ncols;
  vector<T> tmpdata = data;
  data.clear();
  data.insert(data.begin(),
	      tmpdata.begin(),
	      tmpdata.begin() + i*ncols);
  data.insert(data.begin()    + i*ncols,
	      tmpdata.begin() + (i+n)*ncols,
	      tmpdata.begin() + (i+n)*ncols + (nrows-i)*ncols);
  tmpdata.clear();
  ResetBuffer();
}

template <typename T> void PMatrix<T>::DeleteCols(size_t i, size_t n){
  if(i >= ncols || i+n > ncols) return;
  vector<T> tmpdata = data;
  data.clear();
  size_t tmpncols = ncols;
  ncols -= n;
  ndata = nrows * ncols;
  for(size_t m=0; m<nrows; m++){
    data.insert(data.begin()    + m*ncols,
		tmpdata.begin() + m*tmpncols,
		tmpdata.begin() + m*tmpncols  + i);
    data.insert(data.begin()    + m*ncols    + i,
		tmpdata.begin() + m*tmpncols + i + n,
		tmpdata.begin() + m*tmpncols + tmpncols);
  }
  tmpdata.clear();
  ResetBuffer();
}

template <typename T> PMatrix<T>* PMatrix<T>::GetBlock(size_t i, size_t m,
						       size_t j, size_t n){
  PMatrix<T>* block = new PMatrix(mname+"_bl_"+to_string(i)+"_"+to_string(m)+
				  "_"+to_string(j)+"_"+to_string(n),
				  m, n, false);
  block->use_cblas = use_cblas;
  size_t k = j;
  for(size_t r=0; r<m && i<nrows; (r++, i++)){
    k = j;
    for(size_t c=0; c<n && k<ncols; (c++, k++))
      block->data[c+r*n] = data[k+i*ncols];
  }
  return block;
}

template <typename T> void PMatrix<T>::GetBlock(size_t i, size_t j,
						PMatrix<T>* out){
  if(!out) return;
  out->Zero();
  size_t k = j;
  for(size_t r=0; r<out->GetNRows() && i<nrows; (r++, i++)){
    k = j;
    for(size_t c=0; c<out->GetNCols() && k<ncols; (c++, k++))
      out->data[c+r*out->GetNCols()] = data[k+i*ncols];
  }
}

template <typename T> void PMatrix<T>::SetBlock(size_t i, size_t j,
						PMatrix<T> const& obj){
  size_t k = j;
  for(size_t m=0; m<obj.nrows && i<nrows; (m++, i++)){
    k = j;
    for(size_t n=0; n<obj.ncols && k<ncols; (n++, k++))
      data[k+i*ncols] = obj.data[n+m*obj.ncols];
  }
}

template <typename T> void PMatrix<T>::ApplyThreshold(T thresh){
  for(T& v : data) if(abs(v) < abs(thresh)) v = T(0);
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::ApplyLimits(T minval, T maxval){
  for(T& v : data) v = std::min(std::max((isfinite(v)?v:T(0)), minval), maxval);
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::LimitL2Norm(T norm){
  T n = inner_product(data.begin(), data.end(), data.begin(), T(0));
  if(n > norm) (*this) *= norm / n;
}

template <typename T>
vector<set<size_t> >PMatrix<T>::BlockDiagonalize(T thresh){
  if(nrows != ncols) return vector<set<size_t> >(0);
  vector<set<size_t> > blocks(nrows+ncols);
  for(size_t i=0; i<nrows; i++)
    for(size_t j=0; j<ncols; j++)
      if(abs(GetVal(i, j)) > abs(thresh)) blocks[i].insert(j);
  for(size_t i=0; i<ncols; i++)
    for(size_t j=0; j<nrows; j++)
      if(abs(GetVal(j, i)) > abs(thresh)) blocks[nrows+i].insert(j);
  bool empty_intersections = false;
  while(!empty_intersections){
    empty_intersections = true;
    for(size_t i=0; i<blocks.size(); i++)
      for(size_t j=i+1; j<blocks.size(); j++){
	set<size_t> intersection;
	set_intersection(blocks[i].begin(), blocks[i].end(),
			 blocks[j].begin(), blocks[j].end(),
			 inserter(intersection, intersection.begin()));
	if(!intersection.empty()){
	  empty_intersections = false;
	  for(auto it=blocks[j].begin(); it!=blocks[j].end(); it++)
	    blocks[i].insert(*it);
	  blocks.erase(blocks.begin()+j);
	  j --;
	}
      }
  }
  for(size_t i=0; i<blocks.size(); i++)
    if(blocks[i].size() == 0){
      blocks.erase(blocks.begin()+i);
      i --;
    }
  sort(blocks.begin(), blocks.end(),[](const set<size_t>& a,
				       const set<size_t>& b){
    return *(a.begin()) < *(b.begin()); });
  vector<size_t> permutation;
  for(size_t i=0; i<blocks.size(); i++)
    for(auto it=blocks[i].begin(); it!=blocks[i].end(); it++)
      permutation.push_back(*it);
  Permute(permutation);
  ApplyThreshold(thresh);
  return blocks;
}

template <typename T>
pair<T, T> PMatrix<T>::RenormalizeRow(size_t i, T mean, T sigma){
  if(i >= nrows) return make_pair(T(0), T(0));
  size_t n = 0;
  double m = 0.0;
  for(size_t j=0; j<ncols; j++)
    if(!isnan(GetVal(i, j))){
      m += GetVal(i, j);
      n++;
    }
  if(n > 0) m /= n;
  double s = 0.0, t = 0.0;
  for(size_t j=0; j<ncols; j++)
    if(!isnan(GetVal(i, j))){
      t = GetVal(i, j) - m;
      s += t * t;
    }
  if(n > 0) s = sqrt(s / n);
  for(size_t j=0; j<ncols; j++)
    SetVal(i, j, (GetVal(i, j) - m + mean) / (s / sigma));
  return make_pair(m, s);
}

template <typename T>
pair<T, T> PMatrix<T>::RenormalizeCol(size_t i, T mean, T sigma){
  if(i >= ncols) return make_pair(T(0), T(0));
  size_t n = 0;
  double m = 0.0;
  for(size_t j=0; j<nrows; j++)
    if(!isnan(GetVal(j, i))){
      m += (double) GetVal(j, i);
      n ++;
    }
  if(n > 0) m /= n;
  double s = 0.0, t = 0.0;
  for(size_t j=0; j<nrows; j++)
    if(!isnan(GetVal(j, i))){
      t = GetVal(j, i) - m;
      s += t * t;
    }
  if(n > 0) s = sqrt(s / n);
  for(size_t j=0; j<nrows; j++)
    SetVal(j, i, (GetVal(j, i) - m + mean) / (s / sigma));
  return make_pair(m, s);
}
 
template <typename T> void PMatrix<T>::Random(){
  for(size_t i=0; i<ndata; i++) data[i] = uniform_real(rengine);
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::RandomSymmetric(){
  if(nrows != ncols) return;
  for(size_t i=0; i<nrows; i++)
    for(size_t j=i; j<ncols; j++){
      SetVal(i, j, uniform_real(rengine));
      if(i != j) SetVal(j, i, GetVal(i, j));
    }
}

template <typename T> T PMatrix<T>::GetRandom(){
  return T(uniform_real(rengine));
}

template <typename T> void PMatrix<T>::RandomNormal(){
  for(size_t i=0; i<ndata; i++) data[i] = normal_real(rengine);
  brow = -1; bcol = -1;
}

template <typename T> void PMatrix<T>::RandomNormalSymmetric(){
  if(nrows != ncols) return;
  for(size_t i=0; i<nrows; i++)
    for(size_t j=i; j<ncols; j++)
      SetVal(i, j, normal_real(rengine));
  MakeSymmetric(true);
}

template <typename T> T PMatrix<T>::GetRandomNormal(){
  return T(normal_real(rengine));
}

template <typename T> void PMatrix<T>::RandomOrthonormal(size_t fixed_row){
  for(size_t i=0; i<ndata; i++)
    if(i / ncols != fixed_row)
      data[i] = uniform_real(rengine);
  if(fixed_row >= nrows) fixed_row = 0;
  PMatrix<T>* row  = new PMatrix<T>("row",  1, ncols);
  PMatrix<T>* proj = new PMatrix<T>("proj", 1, ncols);
  for(size_t i=0; i<nrows; i++){
    if(i == fixed_row) continue;
    copy(data.begin()+i*ncols, data.begin()+(i+1)*ncols, row->data.begin());
    row->RowProjection(0, fixed_row, *this, proj);
    AddMultRow(i, 0, -1.0, *proj);
    for(size_t j=0; j<i; j++){
      if(j == fixed_row) continue;
      row->RowProjection(0, j, *this, proj);
      AddMultRow(i, 0, -1.0, *proj);
    }
  }
  for(size_t i=0; i<nrows; i++) ScaleRow(i, sqrt(InnerRowProduct(i, i)));
  brow = -1; bcol = -1;
  delete row;
  delete proj;
}

/* places values in V at variable bin width centers defined by the
   quantiles of the distribution of values in V.  any bins larger than
   than (max-min)/max_bins/2, are then split into multiple bins of that
   maximum width.  this results in somewhere between max_bins/2 and max_bins
   variable width bins, depending on V. */
template <typename T> size_t  PMatrix<T>::BinValues(size_t max_bins){
  if(max_bins % 2 != 0) max_bins ++;
  const size_t nbins = max_bins / 2;
  if(nbins == 0) return 0;
  if(data.size() <= max_bins) return data.size();
  // sort the values
  vector<double> v(data.begin(), data.end());
  sort(v.begin(), v.end());
  // construct a cdf from the values
  double norm = 1.0 / v.size();
  vector<double> x({0.0, v[0]}), y({0.0, norm});
  for(size_t i=1; i<v.size(); i++){
    if(v[i] == v[i-1]) y.back() += norm;
    else{
      x.push_back(v[i]);
      y.push_back(y.back()+norm);
    }
  }
  auto minmax = minmax_element(x.begin(), x.end());
  const double min_val = *minmax.first;
  const double max_val = *minmax.second;
  const double max_width = (max_val - min_val) / nbins;
  // get  the quantiles of the cdf
  vector<double> q(nbins+1);
  q[0]     = x.front();
  q[nbins] = x.back();
  size_t j=0;
  for(size_t i=1; i<nbins; i++){
    double p = i / (double) nbins;
    while(y[j] < p) j++;
    q[i] = x[j-1] + (x[j] - x[j-1]) * (p - y[j-1]) / (y[j] - y[j-1]);
  }
  // split bins to keep the width less than the max above
  for(size_t i=0; i<q.size()-1; i++){
    size_t n = (size_t) ((q[i+1] - q[i]) / max_width);
    double dq = (q[i+1] - q[i]) / (n + 1);
    for(size_t j=1; j<=n; j++) q.insert(q.begin() + i + j, q[i] + j * dq);
    i += n;
  }
  // set the values to be those at the bin centers
  for(size_t i=0; i<data.size(); i++){
    double val = data[i];
    size_t j = 0;
    while(q[j] <= val && j < q.size()-1) j ++;
    if(j == q.size()) data[i] = q[j-1];
    else data[i] = (q[j] + q[j-1]) / 2;
  }
  return q.size()-1;
}

template <typename T> bool PMatrix<T>::IsFinite(){
  for(const auto& v : data) if(!isfinite(v)) return false;
  return true;
}

template <typename T> bool PMatrix<T>::IsSymmetric(T thresh){
  if(nrows != ncols) return false;
  for(size_t i=0; i<nrows; i++)
    for(size_t j=i+1; j<ncols; j++)
      if(abs(GetVal(i, j) - GetVal(j, i)) > thresh) return false;
  return true;
}

template <typename T> void PMatrix<T>::MakeSymmetric(bool from_upper){
  if(nrows != ncols) return;
  for(size_t i=0; i<nrows; i++)
    for(size_t j=i+1; j<ncols; j++){
      if(from_upper) SetVal(j, i, GetVal(i, j));
      else SetVal(i, j, GetVal(j, i));
    }
}

template <typename T> void PMatrix<T>::Print(size_t max_size, size_t prec,
					     string lead) const{
  cout.precision(prec);
  cout << scientific;
  cout << lead << mname << "  [" << nrows << ", " << ncols << "]" << endl
       << lead << "[";
  if(ndata == 0) cout << "]";
  for(size_t i=0; i<ndata; i+=ncols){
    if(i == max_size*ncols){
      cout << "... ... ...]";
      break;
    }
    for(size_t j=0; j<ncols; j++){
      if(j == max_size){
	cout << "...";
	break;
      }
      if(data[i+j] >= (T) 0) cout << " ";
      cout << data[i+j];
      if(j < ncols - 1) cout << " ";
    }
    if(i < ndata - ncols) cout << endl << lead << " ";
  }
  if(max_size >= nrows && max_size >= ncols) cout << "]";
  cout << endl; 
}

template <typename T> void PMatrix<T>::Print() const{
  Print(6, 3);
}

template <typename T>
void PMatrix<T>::WriteHDF5(H5::H5File &outfile, string name,
			   size_t chunk_row, size_t chunk_col,
			   int compression) const{
  if(name == "") name = mname;
  hsize_t dims[]    = {nrows, ncols};
  hsize_t maxdims[] = {nrows, ncols};
  hsize_t chunk[]   = {(chunk_row == 0) ? nrows : std::min(nrows, chunk_row),
		       (chunk_col == 0) ? ncols : std::min(ncols, chunk_col)};
  H5::DataSpace dspace(2, dims, maxdims);
  H5::PredType dtype = GetHDF5PredType<T>();
  H5::DSetCreatPropList plist;
  const T val(0);
  plist.setFillValue(dtype, &val);
  plist.setChunk(2, chunk);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  /*T hdata[nrows][ncols];
  for(size_t i=0; i<nrows; i++)
    for(size_t j=0; j<ncols; j++)
    hdata[i][j] = GetVal(i, j);*/
  H5::DataSet dset = outfile.createDataSet(name, dtype, dspace, plist);
  dset.write(&data.front(), dtype);
  dspace.close();
  dset.close();
  plist.close();
}

template <typename T>
void PMatrix<T>::WriteROOT(TFile* outfile, string name) const{
  if(name == "") name = mname;
  TMatrixT<T>* m = new TMatrixT<T>(nrows, ncols, &data.front());
  outfile->WriteObject(m, name.c_str());
  delete m;
}

template <typename T>
void PMatrix<T>::WriteRdata(string outfile, string name,
			    ROOT::R::TRInterface& R){
  if(name == "") name = mname;
  AddToREnv(name, R);
  if(filesystem::exists(outfile)){
    R << ("objs <- load(file.path(\""+outfile+"\"), nenv<-new.env())").c_str();
    R << "nobjs <- length(objs)";
    int nobjs = 0;
    R["nobjs"] >> nobjs;
    vector<string> objs(nobjs);
    R["objs"] >> objs;
    sort(objs.begin(), objs.end());
    auto it = unique(objs.begin(), objs.end());
    objs.erase(it, objs.end());
    it = find(objs.begin(), objs.end(), name);
    if(it != objs.end()) objs.erase(it);
    R["objs"] << objs;
    for(auto const& o : objs) R << (o + " <- nenv$" + o).c_str();
    R << ("save(list=c(objs, \""+name+"\"), file=\""+outfile+"\")").c_str();
  }
  else R << ("save(\"" + name+ "\", file=\"" + outfile + "\")").c_str();
}

template <typename T>
vector<string> PMatrix<T>::AddToREnv(string name, ROOT::R::TRInterface& R){
  Transpose();
  vector<T> row(data.begin(), data.begin()+ncols);
  ROOT::R::TRDataFrame df(ROOT::R::Label["0"]=row);
  for(size_t i=1; i<nrows; i++){
    copy(data.begin()+i*ncols, data.begin()+(i+1)*ncols, row.begin());
    df[to_string(i).c_str()] = row;
  }
  Transpose();
  R[name.c_str()] << df;
  return vector<string>(1, name);
}

template <typename T> bool PMatrix<T>::RowToBuffer(size_t i){
  if(i >= nrows) return false;
  if(i == brow) return true;
  brow = i; bcol = -1;
  i *= ncols;
  for(size_t m=0; m<ncols; m++) buff[m] = data[i+m];
  for(size_t m=ncols; m<buff.size(); m++) buff[m] = T(0);
  return true;
}

template <typename T> bool PMatrix<T>::ColToBuffer(size_t i){
  if(i >= ncols) return false;
  if(i == bcol) return true;
  brow = -1; bcol = i;
  for(size_t m=0; m<nrows; (m++, i+=ncols)) buff[m] = data[i];
  for(size_t m=nrows; m<buff.size(); m++) buff[m] = T(0);
  return true;
}

template <typename T> void PMatrix<T>::ResetBuffer(bool zero){
  size_t n = std::max(nrows, ncols);
  if(buff.size() != n) buff.resize(n);
  if(zero) fill(buff.begin(), buff.end(), T());
  brow = -1; bcol = -1;
}

#endif

#ifndef __BGMatrix__
#define __BGMatrix__

#include "BDMatrix.hh"
#include "GMatrix.hh"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <type_traits>

using namespace std;

template <typename T> class BGMatrix : public BDMatrix<T>{
public:
  BGMatrix();
  BGMatrix(string name, cublasHandle_t h);
  BGMatrix(string name, BGMatrix<T> const& obj);
  BGMatrix(string name, BDMatrix<T> const& obj, cublasHandle_t h);
  BGMatrix(string name, PMatrix<T>& obj, T thresh, cublasHandle_t h);
  virtual ~BGMatrix();

  virtual void ChangeDevLock(bool lock);
  virtual void OpenForAppend();
  virtual void CloseToAppend();
  
  virtual void AppendBlock(PMatrix<T>* block);
  virtual void SortBlocks();
  virtual PMatrix<T>* GetBlock(size_t bindex) const;
  virtual void GetBlock(size_t bindex, PMatrix<T>& block) const;
  virtual void CopyBlock(size_t bindex, PMatrix<T>* block);
  virtual void ClearBlocks();

  bool operator==(BGMatrix<T> const& obj);
  bool operator&(BGMatrix<T>  const& obj) const;

  BGMatrix<T>* operator+(BGMatrix<T> const& obj);
  BGMatrix<T>* operator-(BGMatrix<T> const& obj);
  BGMatrix<T>* operator*(BGMatrix<T> const& obj);
  BGMatrix<T>* operator*(T val);

  BGMatrix<T>& operator+=(BGMatrix<T> const& obj);
  BGMatrix<T>& operator-=(BGMatrix<T> const& obj);
  BGMatrix<T>& operator*=(T val);

  virtual BGMatrix<T>* axpy_oop(string name, T alpha, BGMatrix<T> const& obj);
  virtual void axpy(T alpha, BGMatrix<T> const& obj);
  virtual void RightMult(BGMatrix<T> const& obj, T alpha,
			 BGMatrix<T>*       out, T beta) const;

  virtual T GetVal(size_t index) const;
  virtual T GetVal(size_t row, size_t col) const;
  virtual T GetVal(size_t bindex, size_t brow, size_t bcol) const;
  virtual vector<T>& GetData();
  virtual const vector<T>& GetDataConst() const;
  virtual void SetVal(size_t row, size_t col, T val);
  virtual void SetVal(size_t bindex, size_t brow, size_t bcol, T val);
  virtual void SetData(vector<T>& d);
  virtual pair<size_t, size_t> IndexToBlockIndex(size_t ind) const;
  
  virtual T GetBlockVal(size_t bindex, size_t index) const;
  virtual T GetBlockVal(size_t bindex, size_t row, size_t col) const;
  virtual size_t BlockIndexToIndex(size_t bindex, size_t index) const;

  virtual void SetAll(T val);
  virtual void Zero();

  void Inverse(BGMatrix<T>* inv, double* logdet=NULL) const;
  virtual BGMatrix<T>* Sqrt();
  virtual T Trace();
  virtual T TraceProduct(BGMatrix<T>& obj);
  virtual void Transpose();

  virtual void Random();

  virtual T* GetDeviceData();

  virtual void RightMult(PMatrix<T>& obj, PMatrix<T>* out) const;
  virtual void RightMult(GMatrix<T>& obj, T alpha,
			 GMatrix<T>* out, T beta) const;

  virtual void Print(size_t max_bl, size_t max_size,
		     size_t prec, string lead="") const;
  virtual void Print(size_t max_bl=0) const;

  virtual void WriteHDF5(H5::H5File &outfile, string name="") const;
  virtual void WriteROOT(TFile* outfile,      string name="") const;
  virtual void WriteRdata(string outfile,     string name,
			  ROOT::R::TRInterface& R);
  virtual vector<string> AddToREnv(string name, ROOT::R::TRInterface& R);
  
protected:
  // convenience messages for printing warnings:
  // it is assumed the true case is used to return early from a function
  // and produce a message
  virtual bool IsDevLocked()  const;
  virtual bool IsHostLocked() const;
  virtual bool IsOpen()       const;
  virtual bool IsClosed()     const;
  // management of host to device data
  virtual void AllocateOnDevice();
  virtual void CopyToDevice();
  virtual void ReadFromDevice();
  virtual void FreeOnDevice();
  
  bool closed;
  bool dev_lock;
  cublasHandle_t handle;
  T* dev_data;
  vector<T**> cbs_dev_pntr_arr;
  vector<size_t> cbs_begin;
  vector<size_t> cbs_nblocks;
  vector<size_t> cbs_nrows;
};

template <typename T> BGMatrix<T>::BGMatrix() :
  BDMatrix<T>(){
  closed       = false;
  dev_lock     = true;
  handle       = 0;
  dev_data     = NULL;
}

template <typename T> BGMatrix<T>::BGMatrix(string name, cublasHandle_t h) :
  BDMatrix<T>(name){
  closed       = false;
  dev_lock     = true;
  handle       = h;
  dev_data     = NULL;
}

template <typename T>
BGMatrix<T>::BGMatrix(string name, BGMatrix<T> const& obj) :
  BDMatrix<T>(name, (BDMatrix<T>) obj){
  handle       = obj.handle;
  dev_data     = NULL;
  cbs_begin    = obj.cbs_begin;
  cbs_nblocks  = obj.cbs_nblocks;
  cbs_nrows    = obj.cbs_nrows;
  cbs_dev_pntr_arr.resize(cbs_begin.size());
  if(obj.dev_data){
    closed   = true;
    dev_lock = false;
    AllocateOnDevice();
    CopyToDevice();
  }
  closed       = obj.closed;
  dev_lock     = obj.dev_lock;
}

template <typename T>
BGMatrix<T>::BGMatrix(string name, BDMatrix<T> const& obj, cublasHandle_t h) :
  BDMatrix<T>(name, obj){
  closed       = false;
  dev_lock     = true;
  handle       = h;
  dev_data     = NULL;
}

template <typename T>
BGMatrix<T>::BGMatrix(string name, PMatrix<T>& obj, T thresh, cublasHandle_t h) :
  BDMatrix<T>(name, obj, thresh){
  closed       = false;
  dev_lock     = true;
  handle       = h;
  dev_data     = NULL;
}

template <typename T> BGMatrix<T>::~BGMatrix(){
  dev_lock = false;
  FreeOnDevice();
}

template <typename T> void  BGMatrix<T>::ChangeDevLock(bool lock){
  if(dev_lock == lock) return;
  if(!closed){
    cout << "BGMatrix<T>::ChangeDevLock - object must be closed "
	 << " in order to change the lock, use CloseToAppend()" << endl;
    return;
  }
  if(dev_lock){
    dev_lock = false;
    CopyToDevice();
  }
  else{
    dev_lock = true;
    ReadFromDevice();
  }
}

template <typename T> void BGMatrix<T>::OpenForAppend(){
  if(!closed || IsHostLocked()) return;
  ReadFromDevice();
  dev_lock = false;
  FreeOnDevice();
  dev_lock = true;
  closed = false;
}

template <typename T> void BGMatrix<T>::CloseToAppend(){
  if(closed || IsHostLocked()) return;
  //closed   = true;
  //dev_lock = true;
  SortBlocks();
  size_t nrlast = 0;
  size_t offset = 0;
  for(size_t i=0; i<this->GetNBlocks(); i++){
    if(this->GetBlockRows(i) == nrlast)
      cbs_nblocks.back() ++;
    else{
      cbs_begin.push_back(offset);
      cbs_nblocks.push_back((size_t) 1);
      cbs_nrows.push_back(this->GetBlockRows(i));
      nrlast = this->GetBlockRows(i);
    }
    offset += this->GetBlockNData(i);
  }
  cbs_dev_pntr_arr.resize(cbs_begin.size(), NULL);
  closed = true;
}

template <typename T> void BGMatrix<T>::AppendBlock(PMatrix<T>* block){
  if(IsClosed() || IsHostLocked()) return;
  BDMatrix<T>::AppendBlock(block);
}

template <typename T> void BGMatrix<T>::SortBlocks(){
  if(IsClosed() || IsHostLocked()) return;
  BDMatrix<T>::SortBlocks();
}

template <typename T> PMatrix<T>* BGMatrix<T>::GetBlock(size_t bindex) const{
  if(IsHostLocked()) return NULL;
  return BDMatrix<T>::GetBlock(bindex);
}

template <typename T>
void BGMatrix<T>::GetBlock(size_t bindex, PMatrix<T>& block) const{
  if(IsHostLocked()) return;
  BDMatrix<T>::GetBlock(bindex, block);
}

template <typename T>
void BGMatrix<T>::CopyBlock(size_t bindex, PMatrix<T>* block){
  if(IsHostLocked()) return;
  BDMatrix<T>::CopyBlock(bindex, block);
}

template <typename T> void BGMatrix<T>::ClearBlocks(){
  if(IsClosed() || IsHostLocked()) return;
  BDMatrix<T>::ClearBlocks();
}

template <typename T> bool BGMatrix<T>::operator==(BGMatrix<T> const& obj){
  if(IsHostLocked()) return false;
  return BDMatrix<T>::operator==(static_cast<BDMatrix<T> >(obj));
}

template <typename T>
bool BGMatrix<T>::operator&(BGMatrix<T>  const& obj) const{
  if(IsOpen()) return false;
  return BDMatrix<T>::operator&(static_cast<BDMatrix<T> >(obj));
}

template <typename T>
BGMatrix<T>* BGMatrix<T>::operator+(BGMatrix<T> const& obj){
  if(*this & obj)
    return axpy_oop(this->mname+"_"+obj.GetName()+"_sum", T(1), obj);
  else return NULL;
}

template <typename T>
BGMatrix<T>* BGMatrix<T>::operator-(BGMatrix<T> const& obj){
  if(*this & obj)
    return axpy_oop(this->mname+"_"+obj.GetName()+"_diff", T(-1), obj);
  else return NULL;
}

template <typename T>
BGMatrix<T>* BGMatrix<T>::operator*(BGMatrix<T> const& obj){
  if(IsOpen() || IsDevLocked()) return NULL;
  if(!(*this & obj)) return NULL;
  BGMatrix<T>* out = new BGMatrix(this->mname+"_"+obj.GetName()+"_mult", *this);
  cudaMemset(out->dev_data, 0, sizeof(T)*out->data.size());
  RightMult(obj, T(1), out, T(0));
  return out;
}

template <typename T> BGMatrix<T>* BGMatrix<T>::operator*(T val){
  if(IsOpen() || IsDevLocked()) return NULL;
  BGMatrix<T>* out = new BGMatrix<T>(this->mname+"_scal", *this);
  cublasStatus_t ret;
  if(is_same<T, float>::value)
    ret = cublasSscal(handle, out->data.size(),
		      (float*) &val, (float*) dev_data, 1);
  else if(is_same<T, double>::value)
    ret = cublasDscal(handle, out->data.size(),
		      (double*) &val, (double*) dev_data, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret != CUBLAS_STATUS_SUCCESS){
    cout << "BGMatrix<T>::operator*= cuBLAS error "
	 << cublasGetStatusString(ret) << endl;
    cudaDeviceReset();
  }
  return out;
}

template <typename T>
BGMatrix<T>& BGMatrix<T>::operator+=(BGMatrix<T> const& obj){
  if(IsOpen() || IsDevLocked()) return *this;
  axpy(T(1), obj);
  return *this;
}

template <typename T>
BGMatrix<T>& BGMatrix<T>::operator-=(BGMatrix<T> const& obj){
  if(IsOpen() || IsDevLocked()) return *this;
  axpy(T(-1), obj);
  return *this;
}

template <typename T> BGMatrix<T>& BGMatrix<T>::operator*=(T val){
  if(IsOpen() || IsDevLocked()) return NULL;
  cublasStatus_t ret;
  if(is_same<T, float>::value)
    ret = cublasSscal(handle, this->data.size(),
		      (float*) &val, (float*) dev_data, 1);
  else if(is_same<T, double>::value)
    ret = cublasDscal(handle, this->data.size(),
		      (double*) &val, (double*) dev_data, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret != CUBLAS_STATUS_SUCCESS){
    cout << "BGMatrix<T>::operator*= cuBLAS error "
	 << cublasGetStatusString(ret) << endl;
    cudaDeviceReset();
  }
  return *this;  
}

template <typename T>
BGMatrix<T>* BGMatrix<T>::axpy_oop(string name, T alpha,
				   BGMatrix<T> const& obj){
  if(IsOpen() || IsDevLocked()) return NULL;
  if(!(*this & obj)) return NULL;
  BGMatrix<T>* out = new BGMatrix<T>(name, *this);
  cublasStatus_t ret;
  if(is_same<T, float>::value)
    ret = cublasSaxpy(handle, this->data.size(), (float*) &alpha,
		      (float*) obj.dev_data, 1,  (float*) out->dev_data, 1);
  else if(is_same<T, double>::value)
    ret = cublasDaxpy(handle, this->data.size(), (double*) &alpha,
		      (double*) obj.dev_data, 1, (double*) out->dev_data, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret != CUBLAS_STATUS_SUCCESS){
    cout << "BGMatrix<T>::operator+= cuBLAS error "
	 << cublasGetStatusString(ret) << endl;
    cudaDeviceReset();
  }
  return out;
}

template <typename T>  void BGMatrix<T>::axpy(T alpha, BGMatrix<T> const& obj){
  if(IsOpen() || IsDevLocked()) return;
  cublasStatus_t ret;
  if(is_same<T, float>::value)
    ret = cublasSaxpy(handle, this->data.size(), (float*) &alpha,
		      (float*) obj.dev_data, 1,  (float*) dev_data, 1);
  else if(is_same<T, double>::value)
    ret = cublasDaxpy(handle, this->data.size(), (double*) &alpha,
		      (double*) obj.dev_data, 1, (double*) dev_data, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret != CUBLAS_STATUS_SUCCESS){
    cout << "BGMatrix<T>::operator+= cuBLAS error "
	 << cublasGetStatusString(ret) << endl;
    cudaDeviceReset();
  }
}

template <typename T>
void BGMatrix<T>::RightMult(BGMatrix<T> const& obj, T alpha,
			    BGMatrix<T>*       out, T beta) const{
  if(IsOpen() || IsDevLocked()) return;
  if(!out) return;
  if(!(*this & obj) || !(*this & *out)) return;
  for(size_t i=0; i<cbs_begin.size(); i++){
    const int N = (int) cbs_nblocks[i];
    const int n = (int) cbs_nrows[i];
    cublasStatus_t ret;
    if(is_same<T, float>::value)
      ret = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
			       n, n, n, (float*) &alpha,
			       (float**) cbs_dev_pntr_arr[i], n,
			       (float**) obj.cbs_dev_pntr_arr[i], n,
			       (float*)  &beta,
			       (float**) out->cbs_dev_pntr_arr[i], n,  N);
    else if(is_same<T, double>::value){
      ret = cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
			       n, n, n, (double*) &alpha,
			       (double**) cbs_dev_pntr_arr[i], n,
			       (double**) obj.cbs_dev_pntr_arr[i], n,
			       (double*)  &beta,
			       (double**) out->cbs_dev_pntr_arr[i], n,  N);
    }
    else ret = CUBLAS_STATUS_NOT_SUPPORTED;
    if(ret != CUBLAS_STATUS_SUCCESS){
      cout << "BGMatrix<T>::RightMult cuBLAS error "
	   << cublasGetStatusString(ret) << endl;
      cudaDeviceReset();
    }
  }
  out->Transpose();
}

template <typename T> T BGMatrix<T>::GetVal(size_t index) const{
  if(IsOpen() || IsHostLocked()) return T(0);
  return BDMatrix<T>::GetVal(index);
}

template <typename T> T BGMatrix<T>::GetVal(size_t row, size_t col) const{
  if(IsOpen() || IsHostLocked()) return T(0);
  return BDMatrix<T>::GetVal(row, col);
}

template <typename T>
T BGMatrix<T>::GetVal(size_t bindex, size_t brow, size_t bcol) const{
  if(IsOpen() || IsHostLocked()) return T(0);
  return BDMatrix<T>::GetVal(bindex, brow, bcol);
}

template <typename T> vector<T>& BGMatrix<T>::GetData(){
  if(IsOpen() || IsHostLocked()){ }
  return BDMatrix<T>::GetData();
}

template <typename T> const vector<T>& BGMatrix<T>::GetDataConst() const{
  if(IsOpen() || IsHostLocked()){ }
  return BDMatrix<T>::GetDataConst();
}

template <typename T> void BGMatrix<T>::SetVal(size_t row, size_t col, T val){
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::SetVal(row, col, val);
}

template <typename T>
void BGMatrix<T>::SetVal(size_t bindex, size_t brow, size_t bcol, T val){
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::SetVal(bindex, brow, bcol, val);
}

template <typename T> void BGMatrix<T>::SetData(vector<T>& d){
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::SetData(d);
}

template <typename T>
pair<size_t, size_t> BGMatrix<T>::IndexToBlockIndex(size_t ind) const{
  if(IsOpen()) return make_pair(0, 0);
  return BDMatrix<T>::IndexToBlockIndex(ind);
}

template <typename T>
T BGMatrix<T>::GetBlockVal(size_t bindex, size_t index) const{
  if(IsOpen() || IsHostLocked()) return T(0);
  return BDMatrix<T>::GetBlockVal(bindex, index);
}

template <typename T>
T BGMatrix<T>::GetBlockVal(size_t bindex, size_t row, size_t col) const{
  if(IsOpen() || IsHostLocked()) return T(0);
  return BDMatrix<T>::GetBlockVal(bindex, row, col);  
}

template <typename T>
size_t BGMatrix<T>::BlockIndexToIndex(size_t bindex, size_t index) const{
  if(IsOpen()) return 0;
  return BDMatrix<T>::BlockIndexToIndex(bindex, index);
}

template <typename T> void BGMatrix<T>::SetAll(T val){
  if(IsOpen()) return;
  BDMatrix<T>::SetAll(val);
  if(!dev_lock) CopyToDevice();
}

template <typename T> void BGMatrix<T>::Zero(){
  SetAll(T(0));
}

template <typename T>
void BGMatrix<T>::Inverse(BGMatrix<T>* inv, double* logdet) const{
  if(IsOpen() || IsDevLocked()) return;
  if(!inv) inv = new BGMatrix<T>(this->mname+"_inv", *this);  
  if(!(*this & *inv)) return;
  cudaMemset(inv->dev_data, 0, sizeof(T)*this->GetNData());
  BGMatrix<T>* tmp = new BGMatrix<T>(this->mname+"_tmp", *this);
  tmp->Transpose();
  int* pivot = NULL, *info = NULL;
  cudaMalloc((void**) &pivot, sizeof(int) * this->nrows);
  cudaMalloc((void**) &info,  sizeof(int) * this->GetNBlocks());
  vector<int> info_host(this->GetNBlocks());
  for(size_t i=0; i<cbs_begin.size(); i++){
    cudaMemset(pivot, 0, sizeof(int) * this->nrows);
    cudaMemset(info,  0, sizeof(int) * this->GetNBlocks());
    const int N = (int) cbs_nblocks[i];
    const int n = (int) cbs_nrows[i];
    cublasStatus_t ret;
    if(is_same<T, float>::value)
      ret = cublasSgetrfBatched(handle, n, (float**) tmp->cbs_dev_pntr_arr[i],
				n, pivot, info, N);
    else if(is_same<T, double>::value)
      ret = cublasDgetrfBatched(handle, n, (double**) tmp->cbs_dev_pntr_arr[i],
				n, pivot, info, N);
    else ret = CUBLAS_STATUS_NOT_SUPPORTED;
    if(ret != CUBLAS_STATUS_SUCCESS){
      cout << "GMatrix<T>::Inverse() cuBLAS factor error " << ret << endl;
      cudaDeviceReset();
      cudaFree(pivot); cudaFree(info); delete tmp;
      return;
    }
    cudaMemcpy(&info_host.front(), info, sizeof(int)*N, cudaMemcpyDeviceToHost);
    for(int j=0; j<N; j++){
      if(info_host[j] != 0){
	cout << "BGMatrix<T>::Inverse() factorization failed, "
	     << "matrix may be singular, " << info_host[j] << endl;
	cudaFree(pivot); cudaFree(info); delete tmp;
	return;
      }
    }
    if(is_same<T, float>::value)
      ret = cublasSgetriBatched(handle, n,
				(float**)tmp->cbs_dev_pntr_arr[i], n, pivot,
				(float**)inv->cbs_dev_pntr_arr[i], n, info, N);
    else if(is_same<T, double>::value)
      ret = cublasDgetriBatched(handle, n,
				(double**)tmp->cbs_dev_pntr_arr[i], n, pivot,
				(double**)inv->cbs_dev_pntr_arr[i], n, info, N);
    else ret = CUBLAS_STATUS_NOT_SUPPORTED;
    if(ret != CUBLAS_STATUS_SUCCESS){
      cout << "BGMatrix<T>::Inverse() cuBLAS inverse error " << ret << endl;
      cudaDeviceReset();
      cudaFree(pivot); cudaFree(info); delete tmp;
      return;
    }
    cudaMemcpy(&info_host.front(), info, sizeof(int)*N, cudaMemcpyDeviceToHost);
    for(int j=0; j<N; j++){
      if(info_host[j] != 0){
	cout << "BGMatrix<T>::Inverse() inversion failed, "
	     << "matrix may be singular, " << info_host[j] << endl;
	cudaFree(pivot); cudaFree(info); delete tmp;
	return;
      }
    }  
  }
  cudaFree(pivot); cudaFree(info); delete tmp;
  inv->Transpose();
}

template <typename T> BGMatrix<T>* BGMatrix<T>::Sqrt(){
  if(IsOpen() || IsHostLocked()) return NULL;
  BDMatrix<T>* sq  = BDMatrix<T>::Sqrt();
  BGMatrix<T>* ret = new BGMatrix<T>(this->mname+"_sqrt", *sq, handle);
  delete sq;
  return ret;
}

template <typename T> T BGMatrix<T>::Trace(){
  if(IsOpen()) return T(0);
  if(dev_lock) return BDMatrix<T>::Trace();
  else{
    T tr = T(0);
    T* tmp = (double*) malloc(sizeof(double));
    for(size_t i=0; i<cbs_begin.size(); i++){
      tmp[0] = T(0);
      const int m = (int) (cbs_nrows[i] + 1);
      const int n = (int) (cbs_nblocks[i]*cbs_nrows[i]*cbs_nrows[i]);
      cublasStatus_t ret;
      if(is_same<T, float>::value)
	ret = cublasSasum(handle, n, (float*) (dev_data+cbs_begin[i]),
			  m, (float*) tmp);
      else if(is_same<T, float>::value)
	ret = cublasDasum(handle, n, (double*) (dev_data+cbs_begin[i]),
			  m, (double*) tmp);
      else ret = CUBLAS_STATUS_NOT_SUPPORTED;
      if(ret != CUBLAS_STATUS_SUCCESS){
	cout << "BGMatrix<T>::Trace() cuBLAS error "
	     << cublasGetStatusString(ret) << endl;
	cudaDeviceReset();
	return T(0);
      }
      tr += tmp[0];
    }
    return tr;
  }
}

template <typename T> T BGMatrix<T>::TraceProduct(BGMatrix<T>& obj){
  if(IsOpen() || IsHostLocked()) return T(0);
  return BDMatrix<T>::TraceProduct(static_cast<BDMatrix<T>&>(obj));
}

template <typename T> void BGMatrix<T>::Transpose(){
  if(IsOpen()) return;
  if(dev_lock) BDMatrix<T>::Transpose();
  else{
    T* tmp = NULL;
    cudaMalloc((void**) &tmp, sizeof(T)*this->GetNData());
    cudaMemcpy(tmp, dev_data, sizeof(T)*this->GetNData(),
	       cudaMemcpyDeviceToDevice);
    T alpha = T(1);
    T beta =  T(0);
    int offset = 0;
    for(size_t i=0; i<this->GetNBlocks(); i++){
      const int n = (int) this->bdim[i];
      if(n == 1){
	offset ++;
	continue;
      }
      T* p = dev_data + offset;
      T* t = tmp + offset;
      cublasStatus_t ret;
      if(is_same<T, float>::value)
	ret = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
			  (float*) &alpha, (float*) t, n,
			  (float*) &beta,  NULL, n, (float*) p, n);
      else if(is_same<T, double>::value)
	ret = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
			  (double*) &alpha, (double*) t, n,
			  (double*) &beta,  NULL, n, (double*) p, n); 
      else ret = CUBLAS_STATUS_NOT_SUPPORTED;
      if(ret != CUBLAS_STATUS_SUCCESS){
	cout << "BGMatrix<T>::Transpose() cuBLAS error "
	     << cublasGetStatusString(ret) << endl;
	cudaDeviceReset();
	return;
      }
      offset += n * n;
    }
    cudaFree(tmp);
  }
}

template <typename T> void BGMatrix<T>::Random(){
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::Random();
}

template <typename T> T* BGMatrix<T>::GetDeviceData(){
  if(IsOpen() || IsDevLocked()) return NULL;
  return dev_data;
}

template <typename T> void BGMatrix<T>::RightMult(PMatrix<T>& obj,
						  PMatrix<T>* out) const{
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::RightMult(obj, out);
}

template <typename T> void BGMatrix<T>::RightMult(GMatrix<T>& obj, T alpha,
						  GMatrix<T>* out, T beta)const{
  if(IsOpen() || IsDevLocked()) return;
  if(!out) return;
  if(this->GetNCols() != obj.GetNRows() ||
     this->GetNCols() != out->GetNRows()) return;
  if(obj.GetNCols() != out->GetNCols()) return;
  if(obj.GetNCols() > 1){
    cout << "BGMatrix<T>::RightMult(GMatrix<T>, GMatrix<T>*) is "
	 << "currently only supported for GMatrix column vectors" << endl;
    return;
  }
  if(!obj.GetDeviceData() || !out->GetDeviceData()) return;
  size_t offset = 0;
  for(size_t i=0; i<cbs_begin.size(); i++){
    const int N = (int) cbs_nblocks[i];
    const int n = (int) cbs_nrows[i];
    T** obj_pntr_host = (T**) malloc(sizeof(T*) * N);
    T** out_pntr_host = (T**) malloc(sizeof(T*) * N);
    for(int j=0; j<N; j++){
      obj_pntr_host[j] =  obj.GetDeviceData() + offset + j*n;
      out_pntr_host[j] = out->GetDeviceData() + offset + j*n;
    }
    T** obj_pntr_dev = NULL;
    T** out_pntr_dev = NULL;
    cudaError_t r = cudaMalloc((void**) &obj_pntr_dev, sizeof(T*)*N);
    if(r != cudaSuccess) cout << cudaGetErrorString(r) << endl;
    cudaMalloc((void**) &out_pntr_dev, sizeof(T*)*N);
    cout << obj_pntr_dev << " " << out_pntr_dev << endl;
    cudaMemcpy(obj_pntr_dev, obj_pntr_host,
	       sizeof(T*)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(out_pntr_dev, out_pntr_host,
	       sizeof(T*)*N, cudaMemcpyHostToDevice);
    free(obj_pntr_host);
    free(out_pntr_host);
    cublasStatus_t ret;
    if(is_same<T, float>::value)
      ret = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			       n, 1, n, (float*) &alpha,
			       (float**) cbs_dev_pntr_arr[i], n,
			       (float**) obj_pntr_dev, n,
			       (float*)  &beta,
			       (float**) out_pntr_dev, n,  N);
    else if(is_same<T, double>::value)
      ret = cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			       n, 1, n, (double*) &alpha,
			       (double**) cbs_dev_pntr_arr[i], n,
			       (double**) obj_pntr_dev, n,
			       (double*)  &beta,
			       (double**) out_pntr_dev, n,  N);
    else ret = CUBLAS_STATUS_NOT_SUPPORTED;
    if(ret != CUBLAS_STATUS_SUCCESS){
      cout << "BGMatrix<T>::RightMult cuBLAS error "
	   << cublasGetStatusString(ret) << endl;
      getchar();
      cudaDeviceReset();
    }
    cudaFree(obj_pntr_dev);
    cudaFree(out_pntr_dev);
    offset += N * n;  
  }
}


template <typename T>
void BGMatrix<T>::Print(size_t max_bl, size_t max_size,
			size_t prec, string lead) const{
  if(IsHostLocked()) return;
  BDMatrix<T>::Print(max_bl, max_size, prec, lead);
}

template <typename T> void BGMatrix<T>::Print(size_t max_bl) const{
  Print(max_bl, 6, 3);
}

template <typename T>
void BGMatrix<T>::WriteHDF5(H5::H5File &outfile, string name) const{
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::WriteHDF5(outfile, name);
}

template <typename T>
void BGMatrix<T>::WriteROOT(TFile* outfile, string name) const{
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::WriteROOT(outfile, name);
}

template<typename T>
void BGMatrix<T>::WriteRdata(string outfile, string name,
			     ROOT::R::TRInterface& R){
  if(IsOpen() || IsHostLocked()) return;
  BDMatrix<T>::WriteRdata(outfile, name, R);
}

template <typename T>
vector<string> BGMatrix<T>::AddToREnv(string name, ROOT::R::TRInterface& R){
  if(IsOpen() || IsHostLocked()) return vector<string>(0);
  return BDMatrix<T>::AddToREnv(name, R);
}

template <typename T> bool BGMatrix<T>::IsDevLocked() const{
  if(dev_lock){
    cout << "BGMatrix<T>:: host must be locked for this method" << endl;
    getchar();
    return true;
  }
  else return false;
}

template <typename T> bool BGMatrix<T>::IsHostLocked() const{
  if(!dev_lock){
    cout << "BGMatrix<T>:: device must be locked for this method" << endl;
    getchar();
    return true;
  }
  else return false;
}

template <typename T> bool BGMatrix<T>::IsOpen() const{
  if(!closed){
    cout << "BGMatrix<T>:: matrix must be closed for this method" << endl;
    getchar();
    return true;
  }
  else return false;
}

template <typename T> bool BGMatrix<T>::IsClosed() const{
  if(closed){
    cout << "BGMatrix<T>:: matrix must be open for this method" << endl;
    getchar();
    return true;
  }
  else return false;
}

template <typename T> void BGMatrix<T>::AllocateOnDevice(){
  if(IsOpen() || IsDevLocked()) return;
  if(dev_data) FreeOnDevice();
  cudaError_t ret = cudaMalloc((void**) (&dev_data),
			       this->data.size()*sizeof(T));
  if(ret != cudaSuccess){
    cout << "BGMatrix::AllocateOnDevice() - CUDA error, failed to "
	 << "allocate " << this->data.size()*sizeof(T)
	 << " B on device, " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
    dev_data = NULL;
    return;
  }
  for(size_t i=0; i<cbs_begin.size(); i++){
    const size_t N = cbs_nblocks[i];
    const size_t n = cbs_nrows[i];
    T** cbs_host_pntr_arr = (T**) malloc(sizeof(T*)*N);
    for(size_t j=0; j<N; j++)
      cbs_host_pntr_arr[j] = dev_data + cbs_begin[i] + j*n*n;
    ret = cudaMalloc((void**) &(cbs_dev_pntr_arr[i]), sizeof(T*)*N);
    if(ret != cudaSuccess){
      cout << "BGMatrix::AllocateOnDevice() - CUDA error, failed to "
	   << "allocate " << sizeof(T*)*N
	   << " B on device, " << cudaGetErrorString(ret) << endl;
      cudaDeviceReset();
      cbs_dev_pntr_arr[i] = NULL;
      return;
    }
    ret = cudaMemcpy(cbs_dev_pntr_arr[i], cbs_host_pntr_arr,
		     sizeof(T*)*N, cudaMemcpyHostToDevice);
    if(ret != cudaSuccess){
      cout << "BGMatrix::AllocateOnDevice() - CUDA error, failed to "
	   << "copy " << sizeof(T*)*N
	   << " B on device, " << cudaGetErrorString(ret) << endl;
      cudaDeviceReset();
      cbs_dev_pntr_arr[i] = NULL;
      return;
    }
    free(cbs_host_pntr_arr);
  }
}

template <typename T> void BGMatrix<T>::CopyToDevice(){
  if(IsOpen() || IsDevLocked()) return;
  if(!dev_data) AllocateOnDevice();
  cudaError_t ret = cudaMemcpy(dev_data, &(this->data.front()),
			       this->data.size() * sizeof(T),
			       cudaMemcpyHostToDevice);
  if(ret != cudaSuccess){
    cout << "BGMatrix::CopyToDevice() - CUDA error, failed to "
	 << "copy " << this->data.size()*sizeof(T)
	 << " B to device, " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
  }
}

template <typename T> void BGMatrix<T>::ReadFromDevice(){
  if(IsOpen() || IsHostLocked()) return;
  if(!dev_data){
    cout << "BGMatrix::ReadFromDevice() - no device data available" << endl;
    return;
  }
  cudaError_t ret = cudaMemcpy(&(this->data.front()), dev_data,
			       this->data.size() * sizeof(T),
			       cudaMemcpyDeviceToHost);
  if(ret != cudaSuccess){
    cout << "BGMatrix::ReadFromDevice - CUDA error, failed to "
	 << "read " << this->data.size() * sizeof(T)
	 << " B from device, " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
  }
}

template <typename T> void BGMatrix<T>::FreeOnDevice(){
  if(IsOpen() || IsDevLocked()) return;
  if(!dev_data) return;
  cudaFree(dev_data);
  dev_data = NULL;
  for(size_t i=0; i<cbs_dev_pntr_arr.size(); i++){
    if(cbs_dev_pntr_arr[i]) cudaFree(cbs_dev_pntr_arr[i]);
    cbs_dev_pntr_arr[i] = NULL;
  }
}

#endif

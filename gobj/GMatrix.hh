#ifndef __GMATRIX__
#define __GMATRIX__

#include "PMatrix.hh"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <type_traits>

using namespace std;

template <typename T> class GMatrix : public PMatrix<T>{
public:
  GMatrix();
  GMatrix(string name, size_t nr, size_t nc, bool identity, cublasHandle_t& h);
  GMatrix(string name, GMatrix<T> const& obj);
  GMatrix(string name, PMatrix<T> const& obj, cublasHandle_t& h);
  //GMatrix(string name, H5::H5File const& infile);
  GMatrix(string name, string csv_name, cublasHandle_t& h);
  virtual ~GMatrix();

  virtual bool GetRetainOnDevice() const { return retain_dev; }
  virtual void SetRetainOnDevice(bool retain);
  virtual bool GetNoCopyFromDevice() const { return nocopy_dev; }
  virtual void SetNoCopyFromDevice(bool nocopy){ nocopy_dev = nocopy; }
  
  GMatrix<T>* operator+(PMatrix<T> const& obj);
  GMatrix<T>* operator-(PMatrix<T> const& obj);
  GMatrix<T>* operator*(PMatrix<T>& obj);
  
  GMatrix<T>* operator+(GMatrix<T>& obj);
  GMatrix<T>* operator-(GMatrix<T>& obj);
  GMatrix<T>* operator*(GMatrix<T>& obj);
  GMatrix<T>* operator*(T val);
  
  GMatrix<T>& operator+=(GMatrix<T>& obj);
  GMatrix<T>& operator-=(GMatrix<T>& obj);
  GMatrix<T>& operator*=(T val);

  virtual GMatrix<T>* axpy_oop(string name, T alpha, GMatrix<T>& obj);
  virtual void axpy(T alpha, GMatrix<T>& obj);
  virtual void RightMult(GMatrix<T>& obj, GMatrix<T>* out);
  
  virtual void SetVal(size_t index, T val);
  virtual void SetVal(size_t row, size_t col, T val);

  virtual void SetAll(T val);
  virtual void Zero();
  virtual void AddIdentity(T scale=1);
  virtual void Transpose();
  virtual GMatrix<T>* Inverse();
  virtual GMatrix<T>* BlockwiseInverse(size_t minbl=256);

  virtual void ScaleRow(size_t i, T val);
  virtual void ScaleCol(size_t i, T val);
  virtual void AddMultRow(size_t i, size_t j, T val);
  virtual void AddMultRow(size_t i, size_t j, T val, GMatrix<T>& obj);
  virtual void AddMultCol(size_t i, size_t j, T val);
  virtual void AddMultCol(size_t i, size_t j, T val, GMatrix<T>& obj);
  
  virtual void SetRows(size_t i, size_t n, T val);
  virtual void SetRows(size_t i, size_t j, size_t n, GMatrix<T> const& obj);
  virtual void SetCols(size_t i, size_t n, T val);
  virtual void SetCols(size_t i, size_t j, size_t n, GMatrix<T> const& obj);
  virtual void InsertRows(size_t i, size_t j, size_t n, GMatrix<T> const& obj);
  virtual void InsertCols(size_t i, size_t j, size_t n, GMatrix<T> const& obj);
  virtual void DeleteRows(size_t i, size_t n);
  virtual void DeleteCols(size_t i, size_t n);

  virtual GMatrix<T>* GetBlock(size_t i, size_t m, size_t j, size_t n);

  virtual void Random();

  virtual T* GetDeviceData();
  virtual void AllocateOnDevice();
  virtual void CopyToDevice();
  virtual void ReadFromDevice();
  virtual void FreeOnDevice();

protected:
  virtual void SwapMajorAxis();
  cublasHandle_t handle;
  T* dev_data;
  bool mod_allocate;
  bool mod_host;
  bool mod_dev;
  bool retain_dev;
  bool nocopy_dev;
};

template <typename T> GMatrix<T>::GMatrix() :
  PMatrix<T>(){
  handle       = 0;
  dev_data     = NULL;
  mod_allocate = true;
  mod_host     = true;
  mod_dev      = true;
  retain_dev   = true;
  nocopy_dev   = false;
}

template <typename T> GMatrix<T>::GMatrix(string name, size_t nr, size_t nc,
					  bool identity, cublasHandle_t& h) :
  PMatrix<T>(name, nr, nc, identity){
  handle       = h;
  dev_data     = NULL;
  mod_allocate = true;
  mod_host     = true;
  mod_dev      = true;
  retain_dev   = true;
  nocopy_dev   = false;
}

template <typename T>
GMatrix<T>::GMatrix(string name, GMatrix<T> const& obj) :
  PMatrix<T>(name, (PMatrix<T>) obj){
  handle       = obj.handle;
  dev_data     = NULL;
  mod_allocate = true;
  mod_host     = true;
  mod_dev      = true;
  retain_dev   = obj.retain_dev;
  nocopy_dev   = obj.nocopy_dev;
}

template <typename T>
GMatrix<T>::GMatrix(string name, PMatrix<T> const& obj, cublasHandle_t& h) :
  PMatrix<T>(name, obj){
  handle       = h;
  dev_data     = NULL;
  mod_allocate = true;
  mod_host     = true;
  mod_dev      = true;
  retain_dev   = true;
  nocopy_dev   = false;
}

template <typename T> GMatrix<T>::GMatrix(string name, string csv_name,
					  cublasHandle_t& h) :
  PMatrix<T>(name, csv_name){
  handle       = h;
  dev_data     = NULL;
  mod_allocate = true;
  mod_host     = true;
  mod_dev      = true;
  retain_dev   = true;
  nocopy_dev   = false;
}

template <typename T> GMatrix<T>::~GMatrix(){
  FreeOnDevice();
}

template<typename T> void GMatrix<T>::SetRetainOnDevice(bool retain){
  retain_dev = retain;
  if(!retain_dev) FreeOnDevice();
}

template <typename T> GMatrix<T>* GMatrix<T>::operator+(PMatrix<T> const& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & obj){
    GMatrix<T>* out = new GMatrix(this->mname+"_"+obj.GetName()+"_sum", *this);
    for(size_t i=0; i<this->data.size(); i++) out->data[i] += obj.GetVal(i);
    return out;
  }
  else return new GMatrix<T>();
}

template <typename T> GMatrix<T>* GMatrix<T>::operator-(PMatrix<T> const& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & obj){
    GMatrix<T>* out = new GMatrix(this->mname+"_"+obj.GetName()+"_diff",*this);
    for(size_t i=0; i<this->data.size(); i++) out->data[i] -= obj.GetVal(i);
    return out;
  }
  else return new GMatrix<T>();
}

template <typename T> GMatrix<T>* GMatrix<T>::operator*(PMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) | obj){
    GMatrix<T>* out = new GMatrix<T>(this->mname+"_"+obj.GetName()+"_mult",
				     this->nrows, obj.GetNCols(),false,handle);
    for(size_t i=0; i<obj.GetNCols(); i++)
      for(size_t j=0; j<this->nrows; j++)
	out->data[i+j*this->ncols] = this->InnerProduct(j, i, obj);
    return out;
  }
  else return new GMatrix<T>();
}

template <typename T> GMatrix<T>* GMatrix<T>::operator*(T val){
  GMatrix<T>* out = new GMatrix<T>(this->mname+"_scal", *this);
  (*out) *= val;
  return out;
}

template <typename T> GMatrix<T>* GMatrix<T>::operator+(GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj))
    return axpy_oop(this->mname+"_"+obj.GetName()+"_sum", T(1), obj);
  else return new GMatrix<T>();
}

template <typename T> GMatrix<T>* GMatrix<T>::operator-(GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj))
    return axpy_oop(this->mname+"_"+obj.GetName()+"_diff", T(-1), obj);
  else return new GMatrix<T>();
}

template <typename T> GMatrix<T>* GMatrix<T>::operator*(GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) | obj){
    GMatrix<T>* out = new GMatrix(this->mname+"_"+obj.GetName()+"_mult",
				  this->nrows, obj.ncols, false, handle);
    out->nocopy_dev = nocopy_dev;
    out->retain_dev = retain_dev;
    RightMult(obj, out);
    return out;
  }
  else return new GMatrix<T>();
}

template <typename T> GMatrix<T>& GMatrix<T>::operator+=(GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj))
    axpy(T(1), obj);
  return *this;
}

template <typename T> GMatrix<T>& GMatrix<T>::operator-=(GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj))
    axpy(T(-1), obj);
  return *this;
}

template <typename T> GMatrix<T>& GMatrix<T>::operator*=(T val){
  CopyToDevice();
  T* alpha = (T*) malloc(sizeof(T));
  *alpha = val;
  cublasStatus_t ret;
  mod_dev = true;
  if(is_same<T, float>::value)
    ret = cublasSscal(handle, this->data.size(),
		      (float*) alpha, (float*) dev_data, 1);
  else if(is_same<T, double>::value)
    ret = cublasDscal(handle, this->data.size(),
		      (double*) alpha, (double*) dev_data, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret == CUBLAS_STATUS_SUCCESS){
    if(!nocopy_dev) ReadFromDevice();
  }
  else{
    cout << "GMatrix<T>::operator*= cuBLAS error " << ret << endl;
    cudaDeviceReset();
  }
  return *this;
}

template <typename T> GMatrix<T>* GMatrix<T>::axpy_oop(string name, T val,
						       GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj)){
    GMatrix<T>* out = new GMatrix<T>(name, *this);
    out->CopyToDevice();
    obj.CopyToDevice();
    T* alpha = (T*) malloc(sizeof(T));
    *alpha = val;
    cublasStatus_t ret;
    mod_dev = true;
    if(is_same<T, float>::value)
      ret = cublasSaxpy(handle, this->data.size(), (float*) alpha,
			(float*) obj.dev_data, 1,  (float*) out->dev_data, 1);
    else if(is_same<T, double>::value)
      ret = cublasDaxpy(handle, this->data.size(), (double*) alpha,
			(double*) obj.dev_data, 1, (double*) out->dev_data, 1);
    else ret = CUBLAS_STATUS_NOT_SUPPORTED;
    if(ret == CUBLAS_STATUS_SUCCESS){
      if(!nocopy_dev) out->ReadFromDevice();
    }
    else{
      cout << "GMatrix<T>::operator+= cuBLAS error " << ret << endl;
      cudaDeviceReset();
    }
    if(!obj.retain_dev) obj.FreeOnDevice();
    return out;
  }
  else return new GMatrix<T>();
}

template <typename T> void GMatrix<T>::axpy(T val, GMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj)){
    CopyToDevice();
    obj.CopyToDevice();
    T* alpha = (T*) malloc(sizeof(T));
    *alpha = val;
    cublasStatus_t ret;
    mod_dev = true;
    if(is_same<T, float>::value)
      ret = cublasSaxpy(handle, this->data.size(), (float*) alpha,
			(float*) obj.dev_data, 1,  (float*) dev_data, 1);
    else if(is_same<T, double>::value)
      ret = cublasDaxpy(handle, this->data.size(), (double*) alpha,
			(double*) obj.dev_data, 1, (double*) dev_data, 1);
    else ret = CUBLAS_STATUS_NOT_SUPPORTED;
    if(ret == CUBLAS_STATUS_SUCCESS){
      if(!nocopy_dev) ReadFromDevice();
    }
    else{
      cout << "GMatrix<T>::operator+= cuBLAS error " << ret << endl;
      cudaDeviceReset();
    }
    if(!obj.retain_dev) obj.FreeOnDevice();
  }
}

template <typename T> void GMatrix<T>::RightMult(GMatrix<T>& obj,
						 GMatrix<T>* out){
  if(out){
    if(*dynamic_cast<PMatrix<T>*>(this) | obj){
      CopyToDevice();
      obj.CopyToDevice();
      out->CopyToDevice();
      T* alpha = (T*) malloc(sizeof(T));
      *alpha = (T) 1.0;
      T* beta = (T*) malloc(sizeof(T));
      *beta = (T) 1.0;
      cublasStatus_t ret;
      out->mod_dev = true;
      if(is_same<T, float>::value)
	ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			  this->nrows, obj.ncols, this->ncols, (float*) alpha,
			  (float*) dev_data,      this->nrows,
			  (float*) obj.dev_data,    obj.nrows, (float*) beta,
			  (float*) out->dev_data,  out->nrows);
      else if(is_same<T, double>::value)
	ret = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			  this->nrows, obj.ncols, this->ncols, (double*) alpha,
			  (double*) dev_data,     this->nrows,
			  (double*) obj.dev_data,   obj.nrows, (double*) beta,
			  (double*) out->dev_data, out->nrows);
      else ret = CUBLAS_STATUS_NOT_SUPPORTED;
      free(alpha);
      free(beta);
      if(ret == CUBLAS_STATUS_SUCCESS){
	if(!out->nocopy_dev) out->ReadFromDevice();
      }
      else{
	cout << "GMatrix<T>::operator* cuBLAS error " << ret << endl;
	cudaDeviceReset();
      }
      if(!retain_dev) FreeOnDevice();
      if(!obj.retain_dev) obj.FreeOnDevice();
    }
  }
  else out = (*this) * obj;
}

template <typename T> void GMatrix<T>::SetVal(size_t index, T val){
  PMatrix<T>::SetVal(index, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::SetVal(size_t row, size_t col, T val){
  PMatrix<T>::SetVal(row, col, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::SetAll(T val){
  PMatrix<T>::SetAll(val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::Zero(){
  PMatrix<T>::Zero();
  mod_host = true;
}

template <typename T> void GMatrix<T>::AddIdentity(T scale){
  PMatrix<T>::AddIdentity(scale);
  mod_host = true;
}

template <typename T> void GMatrix<T>::Transpose(){
  PMatrix<T>::Transpose();
  mod_host = true;
}

template <typename T> GMatrix<T>* GMatrix<T>::Inverse(){
  if(this->nrows != this->ncols) return new GMatrix<T>();
  GMatrix<T>* tmp = new GMatrix<T>(this->mname+"_tmp", *this);
  tmp->CopyToDevice();
  T** pntrs = (T**) malloc(sizeof(T*));
  pntrs[0] = tmp->dev_data;
  T** pntrs_dev = NULL;
  cudaMalloc((void**) &pntrs_dev, sizeof(T*));
  cudaMemcpy(pntrs_dev, pntrs, sizeof(T*), cudaMemcpyHostToDevice);
  int *pivot, *info;
  cudaMalloc((void**) &pivot, tmp->nrows*sizeof(int));
  cudaMalloc((void**) &info, sizeof(int));
  cublasStatus_t ret;
  tmp->mod_dev = true;
  if(is_same<T, float>::value)
    ret = cublasSgetrfBatched(tmp->handle, tmp->nrows, (float**) pntrs_dev,
			      tmp->nrows, pivot, info, 1);
  else if(is_same<T, double>::value)
    ret = cublasDgetrfBatched(tmp->handle, tmp->nrows, (double**) pntrs_dev,
			      tmp->nrows, pivot, info, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret != CUBLAS_STATUS_SUCCESS){
    cout << "GMatrix<T>::Inverse() cuBLAS factor error " << ret << endl;
    cudaDeviceReset();
    if(!tmp->retain_dev) tmp->FreeOnDevice();
    free(pntrs); cudaFree(pntrs_dev); cudaFree(pivot); cudaFree(info);
    return tmp;
  }
  int* info_host = new int(0);
  cudaMemcpy(info_host, info, sizeof(int), cudaMemcpyDeviceToHost);
  if(info_host[0] != 0){
    cout << "GMatrix<T>::Inverse() factorization failed, "
	 << "matrix may be singular, " << info_host[0] << endl;
    if(!tmp->retain_dev) tmp->FreeOnDevice();
    free(pntrs); cudaFree(pntrs_dev); cudaFree(pivot); cudaFree(info);
    return tmp;
  }
  GMatrix<T>* inv = new GMatrix<T>(this->mname+"_inv",
				   this->nrows, this->ncols, false, handle);
  inv->nocopy_dev = nocopy_dev;
  inv->retain_dev = retain_dev;
  inv->CopyToDevice();
  T** ipntrs = (T**) malloc(sizeof(T*));
  ipntrs[0] = inv->dev_data;
  T** ipntrs_dev = NULL;
  cudaMalloc((void**) &ipntrs_dev, sizeof(T*));
  cudaMemcpy(ipntrs_dev, ipntrs, sizeof(T*), cudaMemcpyHostToDevice);
  inv->mod_dev = true;
  if(is_same<T, float>::value)
    ret = cublasSgetriBatched(handle, inv->nrows,
			      (float**) pntrs_dev,  tmp->nrows, pivot,
			      (float**) ipntrs_dev, inv->nrows, info, 1);
  else if(is_same<T, double>::value)
    ret = cublasDgetriBatched(handle, inv->nrows,
			      (double**) pntrs_dev,  tmp->nrows, pivot,
			      (double**) ipntrs_dev, inv->nrows, info, 1);
  else ret = CUBLAS_STATUS_NOT_SUPPORTED;
  if(ret == CUBLAS_STATUS_SUCCESS){
    if(!inv->nocopy_dev) inv->ReadFromDevice();
  }
  else{
    cout << "GMatrix<T>::Inverse() cuBLAS inverse error " << ret << endl;
    cudaDeviceReset();
  }
  cudaMemcpy(info_host, info, sizeof(int), cudaMemcpyDeviceToHost);
  if(info_host[0] != 0){
    cout << "GMatrix<T>::Inverse() inversion failed, "
	 << "matrix may be singular, " << info_host[0] << endl;
  }
  if(!inv->retain_dev) inv->FreeOnDevice();
  free(pntrs); cudaFree(pntrs_dev); cudaFree(pivot); cudaFree(info);
  free(ipntrs); cudaFree(ipntrs_dev);
  delete info_host;
  delete tmp;
  return inv;
}
 
template <typename T> GMatrix<T>* GMatrix<T>::BlockwiseInverse(size_t minbl){
  if(this->nrows != this->ncols) return new GMatrix<T>();
  minbl = 1 << (size_t) ceil(log2(minbl));
  if(this->nrows < 2*minbl || minbl < 2) return Inverse();
  const size_t blrows = 1 << ((size_t) ceil(log2(this->nrows)) - 1);
  GMatrix<T>* A = GetBlock(0, blrows, 0, blrows);
  A->retain_dev = true;
  GMatrix<T>* Ainv = NULL;
  if(blrows == minbl){
    PMatrix<T>* pAinv = A->PMatrix<T>::Inverse();
    Ainv = new GMatrix<T>(pAinv->GetName(), *pAinv, handle);
    Ainv->CopyToDevice();
    Ainv->retain_dev = true;
    delete pAinv;
    //Ainv = A->Inverse();
  }
  else Ainv = A->BlockwiseInverse(minbl);
  delete A;
  GMatrix<T>* C = GetBlock(blrows, blrows, 0, blrows);
  C->retain_dev = true;
  GMatrix<T>* CAinv = (*C) * (*Ainv);
  delete C;
  GMatrix<T>* B = GetBlock(0, blrows, blrows, blrows);
  B->retain_dev = true;
  GMatrix<T>* AinvB  = (*Ainv)  * (*B);
  GMatrix<T>* CAinvB = (*CAinv) * (*B);
  delete B;
  GMatrix<T>* D = GetBlock(blrows, blrows, blrows, blrows);
  D->retain_dev = true;
  (*D) -= (*CAinvB);
  delete CAinvB;
  GMatrix<T>* Dinv = NULL;
  if(blrows == minbl){
    D->ReadFromDevice();
    PMatrix<T>* pDinv = D->PMatrix<T>::Inverse();
    Dinv = new GMatrix<T>(pDinv->GetName(), *pDinv, handle);
    Dinv->CopyToDevice();
    Dinv->retain_dev = true;
    delete pDinv;
    //Dinv = D->Inverse();
  }
  else Dinv = D->BlockwiseInverse(minbl);
  delete D;
  GMatrix<T>* inv = new GMatrix<T>(this->mname+"_inv",
				   this->nrows, this->ncols, false, handle);
  inv->mod_allocate = true;
  inv->mod_host = true;
  inv->nocopy_dev = nocopy_dev;
  inv->retain_dev = retain_dev;
  Dinv->ReadFromDevice();
  inv->SetBlock(blrows, blrows, *Dinv);
  GMatrix<T>* AinvBDinv = (*AinvB) * (*Dinv);
  (*AinvBDinv) *= -1.0;
  AinvBDinv->ReadFromDevice();
  inv->SetBlock(0, blrows, *AinvBDinv);
  delete AinvBDinv;
  GMatrix<T>* DinvCAinv = (*Dinv) * (*CAinv);
  delete Dinv;
  delete CAinv;
  (*DinvCAinv) *= -1.0;
  DinvCAinv->ReadFromDevice();
  inv->SetBlock(blrows, 0, *DinvCAinv);
  GMatrix<T>* AinvBDinvCAinv = (*AinvB) * (*DinvCAinv);
  delete AinvB;
  delete DinvCAinv;
  (*Ainv) -= (*AinvBDinvCAinv);
  delete AinvBDinvCAinv;
  Ainv->ReadFromDevice();
  inv->SetBlock(0, 0, *Ainv);
  delete Ainv;
  inv->CopyToDevice();
  return inv;
}

template <typename T> void GMatrix<T>::ScaleRow(size_t i, T val){
  PMatrix<T>::ScaleRow(i, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::ScaleCol(size_t i, T val){
  PMatrix<T>::ScaleCol(i, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::AddMultRow(size_t i, size_t j, T val){
  PMatrix<T>::AddMultRow(i, j, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::AddMultRow(size_t i, size_t j, T val,
						  GMatrix<T>& obj){
  PMatrix<T>::AddMultRow(i, j, val, *dynamic_cast<PMatrix<T>*>(&obj));
  mod_host = true;
}

template <typename T> void GMatrix<T>::AddMultCol(size_t i, size_t j, T val){
  PMatrix<T>::AddMultCol(i, j, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::AddMultCol(size_t i, size_t j, T val,
						  GMatrix<T>& obj){
  PMatrix<T>::AddMultCol(i, j, val, *dynamic_cast<PMatrix<T>*>(&obj));
  mod_host = true;
}

template <typename T> void GMatrix<T>::SetRows(size_t i, size_t n, T val){
  PMatrix<T>::SetRows(i, n, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::SetRows(size_t i, size_t j, size_t n,
					       GMatrix<T> const& obj){
  PMatrix<T>::SetRows(i, j, n, static_cast<PMatrix<T> >(obj));
  mod_host = true;
}

template <typename T> void GMatrix<T>::SetCols(size_t i, size_t n, T val){
  PMatrix<T>::SetCols(i, n, val);
  mod_host = true;
}

template <typename T> void GMatrix<T>::SetCols(size_t i, size_t j, size_t n,
					       GMatrix<T> const& obj){
  PMatrix<T>::SetCols(i, j, n, static_cast<PMatrix<T> >(obj));
}

template <typename T> void GMatrix<T>::InsertRows(size_t i, size_t j, size_t n,
						  GMatrix<T> const& obj){
  PMatrix<T>::InsertRows(i, j, n, static_cast<PMatrix<T> >(obj));
  mod_allocate = true;
  mod_host = true;
}

template <typename T> void GMatrix<T>::InsertCols(size_t i, size_t j, size_t n,
						  GMatrix<T> const& obj){
  PMatrix<T>::InsertCols(i, j, n, static_cast<PMatrix<T> >(obj));
  mod_allocate = true;
  mod_host = true;
}

template <typename T> void GMatrix<T>::DeleteRows(size_t i, size_t n){
  PMatrix<T>::DeleteRows(i, n);
  mod_allocate = true;
  mod_host = true;
}

template <typename T> void GMatrix<T>::DeleteCols(size_t i, size_t n){
  PMatrix<T>::DeleteCols(i, n);
  mod_allocate = true;
  mod_host = true;
}

template <typename T> GMatrix<T>* GMatrix<T>::GetBlock(size_t i, size_t m,
						       size_t j, size_t n){
  PMatrix<T>* pblock = PMatrix<T>::GetBlock(i, m, j, n);
  GMatrix<T>* gblock = new GMatrix<T>(pblock->GetName(), *pblock, handle);
  gblock->mod_allocate = true;
  gblock->mod_host = true;
  gblock->nocopy_dev = nocopy_dev;
  gblock->retain_dev = retain_dev;
  delete pblock;
  return gblock;
}

template <typename T> void GMatrix<T>::Random(){
  PMatrix<T>::Random();
  mod_host = true;
}

template <typename T> T* GMatrix<T>::GetDeviceData(){
  mod_dev = true;
  return dev_data;
}

template <typename T> void GMatrix<T>::AllocateOnDevice(){
  if(dev_data){
    if(mod_allocate) FreeOnDevice();
    else return;
  }
  cudaError_t ret = cudaMalloc((void**) (&dev_data),
			       this->data.size()*sizeof(T));
  if(ret != cudaSuccess){
    cout << "GMatrix::AllocateOnDevice() - CUDA error, failed to "
	 << "allocate " << this->data.size()*sizeof(T)
	 << " B on device, " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
    dev_data = NULL;
    mod_allocate = true;
  }
  mod_allocate = false;
}

template <typename T> void GMatrix<T>::CopyToDevice(){
  if(!dev_data || mod_allocate){
    AllocateOnDevice();
    if(!dev_data){
      mod_host = true;
      return;
    }
  }
  else if(!mod_host) return;
  SwapMajorAxis();
  cudaError_t ret = cudaMemcpy(dev_data, &(this->data.front()),
			       this->data.size() * sizeof(T),
			       cudaMemcpyHostToDevice);
  if(ret != cudaSuccess){
    cout << "GMatrix::CopyToDevice() - CUDA error, failed to "
	 << "copy " << this->data.size()*sizeof(T)
	 << " B to device, " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
    mod_host = true;
  }
  SwapMajorAxis();
  mod_host = false;
}

template <typename T> void GMatrix<T>::ReadFromDevice(){
  if(!dev_data){
    this->Print();
    cout << "GMatrix::ReadFromDevice() - no device data available" << endl;
    mod_dev = true;
    return;
  }
  else if(mod_allocate){
    cout << "GMatrix::ReadFromDevice() - host allocation changed" << endl;
    FreeOnDevice();
    return;
  }
  else if(!mod_dev) return;
  cudaError_t ret = cudaMemcpy(&(this->data.front()), dev_data,
			       this->data.size() * sizeof(T),
			       cudaMemcpyDeviceToHost);
  if(ret == cudaSuccess) SwapMajorAxis();
  else{
    cout << "GMatrix::ReadFromDevice - CUDA error, failed to "
	 << "read " << this->data.size() * sizeof(T)
	 << " B from device, " << cudaGetErrorString(ret) << endl;
    cudaDeviceReset();
    mod_dev = true;
  }
  if(retain_dev) mod_dev = false;
  else FreeOnDevice();
}

template <typename T> void GMatrix<T>::FreeOnDevice(){
  if(!dev_data) return;
  cudaFree(dev_data);
  dev_data = NULL;
  mod_dev = true;
}

template <typename T> void GMatrix<T>::SwapMajorAxis(){
  T tmp;
  size_t offset = 0, cindex = 0;
  for(size_t rindex=0; rindex<this->data.size(); rindex++){
    cindex = this->nrows + offset * (this->nrows + 1);
    rindex += offset;
    offset ++;
    for(; cindex<this->data.size(); cindex+=this->nrows){
      rindex ++;
      tmp = this->data[rindex];
      this->data[rindex] = this->data[cindex];
      this->data[cindex] = tmp;
    }
  }
}

#endif

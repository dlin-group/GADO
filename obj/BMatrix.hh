#ifndef __BMATRIX__
#define __BMATRIX__

#include "PMatrix.hh"
#include <stdlib.h>
#include <string>
#include <cstring>
#include <regex>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <utility>
#include <cstdio>
#include <H5Cpp.h>

using namespace std;
using namespace H5;

template<typename T> class BMatrix : public PMatrix<T>{
public:
  BMatrix();
  BMatrix(string name, size_t nr, size_t nc, size_t br, size_t bc,
	  bool identity, string fname, int compression=0);
  BMatrix(string name, BMatrix<T>& obj,
	  string fname, bool dims_only=false, int compression=0);
  BMatrix(string name, PMatrix<T>& obj, size_t br, size_t bc, string wdir,
	  string fname, bool dims_only=false, int compression=0);
  BMatrix(string name, string fname);
  virtual ~BMatrix();

  bool operator==(BMatrix<T>& obj);

  BMatrix<T>* operator+(BMatrix<T>& obj);
  BMatrix<T>* operator-(BMatrix<T>& obj);
  BMatrix<T>* operator*(BMatrix<T>& obj);

  BMatrix<T>& operator+=(BMatrix<T>& obj);
  BMatrix<T>& operator-=(BMatrix<T>& obj);
  BMatrix<T>& operator*=(T val);

  virtual BMatrix<T>* axpy_oop(string name, T alpha, BMatrix<T>& obj);
  virtual void axpy(T alpha, BMatrix<T>& obj);

  virtual size_t GetNBlockRows() const { return nblrows; }
  virtual size_t GetNBlockCols() const { return nblcols; }
  virtual string GetWorkDir() const { return workdir; }
  virtual void SetWorkDir(string d);
  virtual bool GetRetainFile() const { return retain_file; }
  virtual void SetRetainFile(bool retain){ retain_file = retain; }
  
  virtual void SetAll(T val);
  virtual void Zero();
  virtual void AddIdentity(T scale=1);
  //virtual void Transpose();
  //virtual BMatrix<T>* TransposeOOP();
  virtual BMatrix<T>* Inverse();
  virtual BMatrix<T>* BlockwiseInverse(size_t minbl=0);

  virtual void ScaleRow(size_t i, T val, bool write=true);
  virtual void ScaleCol(size_t i, T val, bool write=true);
  virtual void AddMultRow(size_t i, size_t j, T val,
			  bool write=true, bool tobuff=true);
  virtual void AddMultRow(size_t i, size_t j, T val,
			  BMatrix<T>& obj, bool write=true);
  virtual void AddMultCol(size_t i, size_t j, T val,
			  bool write=true, bool tobuff=true);
  virtual void AddMultCol(size_t i, size_t j, T val,
			  BMatrix<T>& obj, bool write=true);

  virtual size_t GetNRowBlocks() const { return blrcount; }
  virtual size_t GetNColBlocks() const { return blccount; }
  virtual BMatrix<T>* GetBlock(size_t i, size_t m, size_t j, size_t n);
  virtual PMatrix<T>* GetPBlock(size_t i, size_t m, size_t j, size_t n);
  virtual void SetBlock(size_t i, size_t j, BMatrix<T>& obj);
  virtual void SetBlock(size_t i, size_t j, PMatrix<T> const& obj);
  
  virtual void Random();
  
  virtual void BlPrint(size_t max_size, size_t prec, size_t rbl, size_t cbl);
  virtual void Print();

  virtual void SetCurrBlockNoRead(size_t i, size_t j);
  virtual void SetCurrBlockVal(size_t i, size_t j, T val);
  virtual T GetCurrBlockVal(size_t i, size_t j);
  virtual void ReadBlock(size_t i, size_t j, bool breset=true);
  virtual void WriteBlock();
  
protected:
  size_t CurBlockRowOffset();
  size_t CurBlockColOffset();
  size_t CurBlockNRows();
  size_t CurBlockNCols();
  size_t CurSubBlockRowOffset();
  size_t CurSubBlockColOffset();
  size_t CurSubBlockNRows();
  size_t CurSubBlockNCols();
  pair<size_t,size_t> BlockIndexFromIndex(size_t i, size_t j,bool breset=true);
  size_t SubBlockRowFromIndex(size_t i, bool breset=true);
  size_t SubBlockColFromIndex(size_t i, bool breset=true);
  void SetBlockIndex(size_t i, size_t j);
  //void ReadBlock(size_t i, size_t j, bool breset=true);
  void ReadSubBlockRows(size_t i, bool breset=true);
  void ReadSubBlockCols(size_t i, bool breset=true);
  //void WriteBlock();
  void WriteSubBlockRows();
  void WriteSubBlockCols();
  
  size_t nblrows;
  size_t nblcols;
  size_t blrcount;
  size_t blccount;
  size_t sblrcount;
  size_t sblccount;
  size_t currblock;
  size_t curcblock;
  size_t currsblock;
  size_t curcsblock;
  PMatrix<T>* block;
  PMatrix<T>* sblrows;
  PMatrix<T>* sblcols;
  string workdir;
  H5File* datafile;
  DataSet* dataset;
  DataType datatype;
  DataSpace filespace;
  int compression;
  string filename;
  bool retain_file;
};

template <typename T> BMatrix<T>::BMatrix() : PMatrix<T>(){
  nblrows     = 0;
  nblcols     = 0;
  blrcount    = 0;
  blccount    = 0;
  sblrcount   = 0;
  sblccount   = 0;
  currblock   = -1;
  curcblock   = -1;
  currsblock  = -1;
  curcsblock  = -1;
  block       = NULL;
  sblrows     = NULL;
  sblcols     = NULL;
  workdir     = "";
  datafile    = NULL;
  dataset     = NULL;
  compression = 0;
  filename    = "";
  retain_file = false;
}

template <typename T> BMatrix<T>::BMatrix(string name,
					  size_t nr, size_t nc,
					  size_t br, size_t bc,
					  bool identity, string fname,
					  int compression) :
  BMatrix<T>(){
  br = std::min(nr, br);
  bc = std::min(nc, bc);
  this->mname = name;
  this->nrows = nr;
  this->ncols = nc;
  nblrows     = br;
  nblcols     = bc;
  blrcount    = ceil(this->nrows/(double)nblrows);
  blccount    = ceil(this->ncols/(double)nblcols);
  block   = new PMatrix<T>(name+"_block", br, bc, identity);
  sblrows = new PMatrix<T>(name+"_rows",
			   std::max((size_t)1, br*bc/nc), nc, false);
  sblcols = new PMatrix<T>(name+"_cols",
			   nr, std::max((size_t)1, br*bc/nr), false);
  sblrcount = ceil(nblrows/(double)sblrows->GetNRows());
  sblccount = ceil(nblcols/(double)sblcols->GetNCols());
  filename = workdir + fname;
  if(filename.size() > 255){
    size_t trunc = filename.size() - 255;
    filename = filename.substr(0, 126) + "_" + filename.substr(127+trunc, 128);
  }
  datafile  = new H5File(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dims[2] = {(hsize_t) this->nrows, (hsize_t) this->ncols};
  hsize_t chunk[2] = {(hsize_t) nblrows, (hsize_t) nblcols};
  filespace = DataSpace(2, dims);
  if(is_same<T, float>::value)       datatype=PredType::NATIVE_FLOAT;
  else if(is_same<T, double>::value) datatype=PredType::NATIVE_DOUBLE;
  else if(is_same<T, int>::value)    datatype=PredType::NATIVE_INT;
  DSetCreatPropList plist;
  T val(0);
  plist.setFillValue(datatype, &val);
  plist.setChunk(2, chunk);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  dataset = new DataSet(datafile->createDataSet(this->mname, datatype,
						filespace, plist));
  if(identity) AddIdentity(T(1));
}

template <typename T> BMatrix<T>::BMatrix(string name, BMatrix<T>& obj,
					  string fname, bool dims_only,
					  int compression) :
  BMatrix<T>(){
  this->mname = name;
  this->nrows = obj.nrows;
  this->ncols = obj.ncols;
  nblrows     = obj.nblrows;
  nblcols     = obj.nblcols;
  blrcount    = obj.blrcount;
  blccount    = obj.blccount;
  sblrcount   = obj.sblrcount;
  sblccount   = obj.sblccount;
  block       = new PMatrix<T>(name+"_block", *obj.block);
  sblrows     = new PMatrix<T>(name+"_rows",  *obj.sblrows);
  sblcols     = new PMatrix<T>(name+"_cols",  *obj.sblcols);
  workdir     = obj.workdir;
  filename = workdir + fname;
  if(filename.size() > 255){
    size_t trunc = filename.size() - 255;
    filename = filename.substr(0, 126) + "_" + filename.substr(127+trunc, 128);
  }
  datafile    = new H5File(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dims[2] = {(hsize_t) this->nrows, (hsize_t) this->ncols};
  hsize_t chunk[2] = {(hsize_t) nblrows, (hsize_t) nblcols};
  filespace = DataSpace(2, dims);
  if(is_same<T, float>::value)       datatype=PredType::NATIVE_FLOAT;
  else if(is_same<T, double>::value) datatype=PredType::NATIVE_DOUBLE;
  else if(is_same<T, int>::value)    datatype=PredType::NATIVE_INT;
  DSetCreatPropList plist;
  T val(0);
  plist.setFillValue(datatype, &val);
  plist.setChunk(2, chunk);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  dataset = new DataSet(datafile->createDataSet(this->mname, datatype,
						filespace, plist));
  if(dims_only){
    currblock  = -1; curcblock  = -1; currsblock = -1; curcsblock = -1;
  }
  else{
    for(size_t icrb=0; icrb<blrcount; icrb++){
      SetBlockIndex(icrb, -1);
      obj.SetBlockIndex(icrb, -1);
      currsblock = 0;
      for(size_t icrsb=0; icrsb<sblrcount; (currsblock++, icrsb++)){
	obj.ReadSubBlockRows(icrsb);
	copy(obj.sblrows->GetDataConst().begin(),
	     obj.sblrows->GetDataConst().end(), sblrows->GetData().begin());
	WriteSubBlockRows();
      }
    }
    currsblock --;
  }
}

template <typename T> BMatrix<T>::BMatrix(string name, PMatrix<T>& obj,
					  size_t br, size_t bc, string wdir,
					  string fname, bool dims_only,
					  int compression) :
  BMatrix<T>(){
  this->mname = name;
  this->nrows = obj.GetNRows();
  this->ncols = obj.GetNCols();
  nblrows     = br;
  nblcols     = bc;
  blrcount    = ceil(this->nrows/(double)nblrows);
  blccount    = ceil(this->ncols/(double)nblcols);
  block   = new PMatrix<T>(name+"_block", br, bc, false);
  sblrows = new PMatrix<T>(name+"_rows", std::max((size_t)1,br*bc/this->ncols),
			   this->ncols, false);
  sblcols = new PMatrix<T>(name+"_cols", this->nrows,
			   std::max((size_t)1, br*bc/this->nrows), false);
  sblrcount = ceil(nblrows/(double)sblrows->GetNRows());
  sblccount = ceil(nblcols/(double)sblcols->GetNCols());
  workdir     = wdir;
  filename = workdir + fname;
  if(filename.size() > 255){
    size_t trunc = filename.size() - 255;
    filename = filename.substr(0, 126) + "_" + filename.substr(127+trunc, 128);
  }
  datafile    = new H5File(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dims[2] = {(hsize_t) this->nrows, (hsize_t) this->ncols};
  hsize_t chunk[2] = {(hsize_t) nblrows, (hsize_t) nblcols};
  filespace = DataSpace(2, dims);
  if(is_same<T, float>::value)       datatype=PredType::NATIVE_FLOAT;
  else if(is_same<T, double>::value) datatype=PredType::NATIVE_DOUBLE;
  else if(is_same<T, int>::value)    datatype=PredType::NATIVE_INT;
  DSetCreatPropList plist;
  T val(0);
  plist.setFillValue(datatype, &val);
  plist.setChunk(2, chunk);
  plist.setDeflate(std::max(0, std::min(9, compression)));
  dataset = new DataSet(datafile->createDataSet(this->mname, datatype,
						filespace, plist));
  if(dims_only){
    currblock  = -1; curcblock  = -1; currsblock = -1; curcsblock = -1;
  }
  else{
    for(size_t icrb=0; icrb<blrcount; icrb++){
      SetBlockIndex(icrb, -1);
      currsblock = 0;
      size_t offset = icrb * sblrcount;
      for(size_t icrsb=0; icrsb<sblrcount; (currsblock++, icrsb++)){
	copy(obj.GetDataConst().begin()+sblrows->GetNData()*(offset+icrsb),
	     obj.GetDataConst().begin()+sblrows->GetNData()*(offset+icrsb+1),
	     sblrows->GetData().begin());
	WriteSubBlockRows();
      }
    }
    currsblock --;
  }
}

template <typename T> BMatrix<T>::BMatrix(string name, string fname) :
  BMatrix<T>(){
  this->mname = name;
  filename = workdir + fname;
  if(filename.size() > 255){
    size_t trunc = filename.size() - 255;
    filename = filename.substr(0, 126) + "_" + filename.substr(127+trunc, 128);
  }
  if(is_same<T, float>::value)       datatype=PredType::NATIVE_FLOAT;
  else if(is_same<T, double>::value) datatype=PredType::NATIVE_DOUBLE;
  else if(is_same<T, int>::value)    datatype=PredType::NATIVE_INT;
  datafile  = new H5File(filename.c_str(), H5F_ACC_RDWR);
  dataset   = new DataSet(datafile->openDataSet(this->mname));
  filespace = dataset->getSpace();
  assert(filespace.getSimpleExtentNdims() == 2);
  hsize_t dims[2]  = {0, 0};
  hsize_t chunk[2] = {0, 0};
  filespace.getSimpleExtentDims(dims);
  this->nrows = dims[0];
  this->ncols = dims[1];
  DSetCreatPropList plist = dataset->getCreatePlist();
  plist.getChunk(2, chunk);
  nblrows = std::min((size_t) chunk[0], this->nrows);
  nblcols = std::min((size_t) chunk[1], this->ncols);
  blrcount    = ceil(this->nrows/(double)nblrows);
  blccount    = ceil(this->ncols/(double)nblcols);
  block   = new PMatrix<T>(name+"_block", nblrows, nblcols, false);
  sblrows = new PMatrix<T>(name+"_rows",
			   std::max((size_t)1,
				    nblrows*nblcols/this->ncols),
			   this->ncols, false);
  sblcols = new PMatrix<T>(name+"_cols", this->nrows,
			   std::max((size_t)1,
				    nblrows*nblcols/this->nrows), false);
  sblrcount = ceil(nblrows/(double)sblrows->GetNRows());
  sblccount = ceil(nblcols/(double)sblcols->GetNCols());
}

template <typename T> BMatrix<T>::~BMatrix(){
  if(block)      delete block;
  if(sblrows)    delete sblrows;
  if(sblcols)    delete sblcols;
  if(dataset)    delete dataset;
  if(datafile)   delete datafile;
  if(!retain_file) remove(filename.c_str());
}

// rewrite comparison ops with block size checks

template <typename T> bool BMatrix<T>::operator==(BMatrix<T>& obj){
  if(!(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj)))
    return false;
  for(size_t icrb=0; icrb<blrcount; icrb++){
    SetBlockIndex(icrb, -1);
    obj.SetBlockIndex(icrb, -1);
    for(size_t icrsb=0; icrsb<sblrcount; icrsb++){
      ReadSubBlockRows(icrsb);
      obj.ReadSubBlockRows(icrsb);
      for(size_t i=0; i<sblrows->GetNData(); i++)
	if(!(sblrows->GetVal(i) == obj.sblrows->GetVal(i))) return false;
    }
  }
  return true;
}

template <typename T> BMatrix<T>* BMatrix<T>::operator+(BMatrix<T>& obj){
  return axpy_oop(this->mname+"_"+obj.GetName()+"_sum", T(1), obj);
}

template <typename T> BMatrix<T>* BMatrix<T>::operator-(BMatrix<T>& obj){
  return axpy_oop(this->mname+"_"+obj.GetName()+"_diff", T(-1), obj);
}

template <typename T> BMatrix<T>* BMatrix<T>::operator*(BMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj)){
    string name = this->mname + "_" + obj.GetName() + "_mult";
    BMatrix<T>* out = new BMatrix(name, this->nrows, obj.GetNCols(),
				  nblrows, obj.nblcols, false,
				  workdir+name+".h5");
    PMatrix<T>* tmp = NULL;
    for(size_t iccb=0; iccb<obj.blccount; iccb++){
      obj.SetBlockIndex(-1, iccb);
      for(size_t icrb=0; icrb<blrcount; icrb++){
	SetBlockIndex(icrb, -1);
	out->SetBlockIndex(icrb, iccb);
	for(size_t icrsb=0; icrsb<obj.sblccount; icrsb++){
	  obj.ReadSubBlockCols(icrsb);
	  for(size_t jcrsb=0; jcrsb<sblrcount; jcrsb++){
	    ReadSubBlockRows(jcrsb);
	    if(tmp) sblrows->RightMult(*obj.sblcols, tmp);
	    else tmp = (*sblrows) * (*obj.sblcols);
	    size_t m = currsblock * sblrows->GetNRows();
	    for(size_t i=0; i<sblrows->GetNRows(); (i++, m++)){
	      size_t n = obj.curcsblock * obj.sblcols->GetNCols();
	      for(size_t j=0; j<obj.sblcols->GetNCols(); (j++, n++)){
		out->block->SetVal(m, n, tmp->GetVal(i, j));
	      }
	    }
	    delete tmp;
	  }
	}
	out->WriteBlock();
      }
    }
    delete tmp;
    return out;
  }
  else return new BMatrix<T>();
}

template <typename T>
BMatrix<T>& BMatrix<T>::operator+=(BMatrix<T>& obj){
  axpy(T(1), obj);
  return *this;
}

template <typename T>
BMatrix<T>& BMatrix<T>::operator-=(BMatrix<T>& obj){
  axpy(T(-1), obj);
  return *this;
}

template <typename T> BMatrix<T>& BMatrix<T>::operator*=(T val){
  for(size_t icrb=0; icrb<blrcount; icrb++){
    SetBlockIndex(icrb, -1);
    for(size_t icrsb=0; icrsb<sblrcount; icrsb++){
      ReadSubBlockRows(icrsb);
      (*sblrows) *= val;
      WriteSubBlockRows();
    }
  }
  return *this;
}

template <typename T>
BMatrix<T>* BMatrix<T>::axpy_oop(string name, T alpha, BMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj)){
    BMatrix<T>* out = new BMatrix(name, *this, name+".h5", true);
    for(size_t icrb=0; icrb<blrcount; icrb++){
      SetBlockIndex(icrb, -1);
      obj.SetBlockIndex(icrb, -1);
      out->SetBlockIndex(icrb, -1);
      out->currsblock = 0;
      for(size_t icrsb=0; icrsb<sblrcount; (icrsb++, out->currsblock++)){
	ReadSubBlockRows(icrsb);
	obj.ReadSubBlockRows(icrsb);
	copy(sblrows->GetDataConst().begin(),
	     sblrows->GetDataConst().end(), out->sblrows->GetData().begin());
	out->sblrows->axpy(alpha, *obj.sblrows);
	out->WriteSubBlockRows();
      }
    }
    out->currsblock --;
    return out;
  }
  else return new BMatrix<T>();
}

template <typename T> void BMatrix<T>::axpy(T alpha, BMatrix<T>& obj){
  if(*dynamic_cast<PMatrix<T>*>(this) & static_cast<PMatrix<T> >(obj)){
    for(size_t icrb=0; icrb<blrcount; icrb++){
      SetBlockIndex(icrb, -1);
      obj.SetBlockIndex(icrb, -1);
      for(size_t icrsb=0; icrsb<sblrcount; icrsb++){
	ReadSubBlockRows(icrsb);
	obj.ReadSubBlockRows(icrsb);
	sblrows->axpy(alpha, *obj.sblrows);
	WriteSubBlockRows();
      }
    }
  }
}

template <typename T> void BMatrix<T>::SetWorkDir(string dir){
  workdir = dir;
  if(workdir.size() > 0) if(workdir[workdir.size()-1] != '/') workdir += "/";
}

template <typename T> void BMatrix<T>::SetAll(T val){
  for(size_t icrb=0; icrb<blrcount; icrb++){
    SetBlockIndex(icrb, -1);
    for(size_t icrsb=0; icrsb<sblrcount; icrsb++){
      ReadSubBlockRows(icrsb);
      fill(sblrows->GetData().begin(), sblrows->GetData().end(), val);
      WriteSubBlockRows();
    }
  }
}

template <typename T> void BMatrix<T>::Zero(){
  SetAll(T(0));
}

template <typename T> void BMatrix<T>::AddIdentity(T scale){
  size_t index;
  for(size_t icrb=0; icrb<blrcount; icrb++){
    SetBlockIndex(icrb, -1);
    for(size_t icrsb=0; icrsb<sblrcount; icrsb++){
      ReadSubBlockRows(icrsb, true);
      index = CurSubBlockRowOffset();
      for(size_t j=0; j<CurSubBlockNRows(); (j++, index++))
	sblrows->SetVal(j, index, scale + sblrows->GetVal(j, index));
      WriteSubBlockRows();
    }
  }
}

template <typename T> BMatrix<T>* BMatrix<T>::Inverse(){
  if(this->nrows != this->ncols) return new BMatrix<T>();
  BMatrix<T>* tmp = new BMatrix<T>(this->mname+"_tmp", *this,
				   this->mname+"_tmp.h5");
  BMatrix<T>* inv = new BMatrix<T>(this->mname+"_inv",
				   this->nrows, this->ncols, nblrows, nblcols,
				   true, workdir+this->mname+"_inv.h5");
  PMatrix<T>* tbuff = new PMatrix<T>("tmp0", *tmp->sblrows);
  PMatrix<T>* ibuff = new PMatrix<T>("tmp1", *inv->sblrows);
  for(size_t icrb=0; icrb<tmp->blrcount; icrb++){
    for(size_t icrsb=0; icrsb<tmp->sblrcount; icrsb++){
      tmp->SetBlockIndex(icrb, -1);
      inv->SetBlockIndex(icrb, -1);
      tmp->ReadSubBlockRows(icrsb);
      inv->ReadSubBlockRows(icrsb);
      size_t offset = tmp->CurSubBlockRowOffset();
      size_t nrows  = tmp->CurSubBlockNRows();
      for(size_t i=0; i<nrows; i++){
	T val = tmp->sblrows->GetVal(i, offset+i);
	tmp->sblrows->ScaleRow(i, val);
	inv->sblrows->ScaleRow(i, val);
	for(size_t j=0; j<nrows; j++){
	  if(i == j) continue;
	  val = -tmp->sblrows->GetVal(j, offset+i);
	  tmp->sblrows->AddMultRow(j, i, val);
	  inv->sblrows->AddMultRow(j, i, val);
	}
      }
      tmp->WriteSubBlockRows();
      inv->WriteSubBlockRows();
      copy(tmp->sblrows->GetDataConst().begin(),
	   tmp->sblrows->GetDataConst().end(), tbuff->GetData().begin());
      copy(inv->sblrows->GetDataConst().begin(),
	   inv->sblrows->GetDataConst().end(), ibuff->GetData().begin());
      for(size_t jcrb=0; jcrb<tmp->blrcount; jcrb++){
	tmp->SetBlockIndex(jcrb, -1);
	inv->SetBlockIndex(jcrb, -1);
	for(size_t jcrsb=0; jcrsb<tmp->sblrcount; jcrsb++){
	  if(icrb == jcrb && icrsb == jcrsb) continue;
	  cout << icrb << " " << icrsb << " " << jcrb << " " << jcrsb << endl;
	  tmp->ReadSubBlockRows(jcrsb);
	  inv->ReadSubBlockRows(jcrsb);
	  for(size_t i=0; i<nrows; i++)
	    for(size_t j=0; j<nrows; j++){
	      T val = -tmp->sblrows->GetVal(i, offset+j);
	      tmp->sblrows->AddMultRow(i, j, val, *tbuff);
	      inv->sblrows->AddMultRow(i, j, val, *ibuff);
	    }
	  tmp->WriteSubBlockRows();
	  inv->WriteSubBlockRows();
	}
      }
    }
  }
  delete tbuff;
  delete ibuff;
  delete tmp;
  return inv;
}

// add getblock with out as argument, use for A, B, C, D
template <typename T> BMatrix<T>* BMatrix<T>::BlockwiseInverse(size_t minbl){
  if(this->nrows != this ->ncols) return new BMatrix<T>();
  if(minbl == 0) minbl = nblrows;
  minbl = 1 << (size_t) ceil(log2(nblrows));
  if(this->nrows < 2*minbl || minbl < 2){
    PMatrix<T>* tmp = GetPBlock(0, this->nrows, 0, this->ncols);
    PMatrix<T>* pinv = tmp->Inverse();
    delete tmp;
    BMatrix<T>* binv = new BMatrix<T>(pinv->GetName(), *pinv,
				      nblrows, nblcols, workdir,
				      pinv->GetName()+".h5");
    delete pinv;
    return binv;
  }
  const size_t blrows = 1 << ((size_t) ceil(log2(this->nrows)) - 1);
  BMatrix<T>* A = GetBlock(0, blrows, 0, blrows);
  BMatrix<T>* Ainv = NULL;
  if(blrows == minbl){
    PMatrix<T>* tmp = A->GetPBlock(0, A->nrows, 0, A->ncols);
    PMatrix<T>* pinv = tmp->Inverse();
    delete tmp;
    Ainv = new BMatrix<T>(pinv->GetName(), *pinv,
			  A->nblrows, A->nblcols,
			  workdir,pinv->GetName()+".h5");
    BMatrix<T>* t = (*A) * (*Ainv);
    delete t;
  }
  else Ainv = A->BlockwiseInverse(minbl);
  delete A;
  BMatrix<T>* C = GetBlock(blrows, blrows, 0, blrows);
  BMatrix<T>* CAinv = (*C) * (*Ainv);
  delete C;
  BMatrix<T>* B = GetBlock(0, blrows, blrows, blrows);
  BMatrix<T>* AinvB  = (*Ainv)  * (*B);
  BMatrix<T>* CAinvB = (*CAinv) * (*B);
  delete B;
  BMatrix<T>* D = GetBlock(blrows, blrows, blrows, blrows);
  (*D) -= (*CAinvB);
  delete CAinvB;
  BMatrix<T>* Dinv = NULL;
  if(blrows == minbl){
    PMatrix<T>* tmp = D->GetPBlock(0, D->nrows, 0, D->ncols);
    PMatrix<T>* pinv = tmp->Inverse();
    delete tmp;
    Dinv = new BMatrix<T>(pinv->GetName(), *pinv,
			  D->nblrows, D->nblcols,
			  workdir, pinv->GetName()+".h5");
  }
  else Dinv = D->BlockwiseInverse(minbl);
  delete D;
  BMatrix<T>* inv = new BMatrix<T>(this->mname+"_inv",
				   this->nrows, this->ncols, nblrows, nblcols,
				   false, workdir+this->mname+"_inv.h5");
  inv->SetWorkDir(workdir);
  inv->SetBlock(blrows, blrows, *Dinv);
  BMatrix<T>* AinvBDinv = (*AinvB) * (*Dinv);
  (*AinvBDinv) *= -1.0;
  inv->SetBlock(0, blrows, *AinvBDinv);
  delete AinvBDinv;
  BMatrix<T>* DinvCAinv = (*Dinv) * (*CAinv);
  delete Dinv;
  delete CAinv;
  (*DinvCAinv) *= -1.0;
  inv->SetBlock(blrows, 0, *DinvCAinv);
  BMatrix<T>* AinvBDinvCAinv = (*AinvB) * (*DinvCAinv);
  delete AinvB;
  delete DinvCAinv;
  (*Ainv) -= (*AinvBDinvCAinv);
  delete AinvBDinvCAinv;
  inv->SetBlock(0, 0, *Ainv);
  delete Ainv;
  return inv;
}

template <typename T> void BMatrix<T>::ScaleRow(size_t i, T val, bool write){
  if(i >= this->nrows) return;
  sblrows->ScaleRow(SubBlockRowFromIndex(i), val);
  if(write) WriteSubBlockRows();
}

template <typename T> void BMatrix<T>::ScaleCol(size_t i, T val, bool write){
  if(i >= this->ncols) return;
  sblcols->ScaleCol(SubBlockColFromIndex(i), val);
  if(write) WriteSubBlockCols();
}

template <typename T>
void BMatrix<T>::AddMultRow(size_t i, size_t j, T val, bool write,bool tobuff){
  if(i >= this->nrows || j >= this->nrows) return;
  if(tobuff) sblrows->RowToBuffer(SubBlockRowFromIndex(j, false));
  sblrows->AddMultRowBuffer(SubBlockRowFromIndex(i, false), val);
  if(write) WriteSubBlockRows();
}

template <typename T>
void BMatrix<T>::AddMultRow(size_t i, size_t j, T val,
			    BMatrix<T>& obj, bool write){
  if(i >= this->nrows || j >= obj.GetNRows() ||
     this->ncols != obj.GetNCols()) return;
  sblrows->AddMultRow(SubBlockRowFromIndex(i, false),
		      obj.SubBlockRowFromIndex(j, false), val, *obj.sblrows);
  if(write) WriteSubBlockRows();
}

template <typename T>
void BMatrix<T>::AddMultCol(size_t i, size_t j, T val, bool write,bool tobuff){
  if(i >= this->ncols || j >= this->ncols) return;
  if(tobuff) sblcols->ColToBuffer(SubBlockColFromIndex(j, false));
  sblcols->AddMultColBuffer(SubBlockColFromIndex(i, false), val);
  if(write) WriteSubBlockRows();
}

template <typename T>
void BMatrix<T>::AddMultCol(size_t i, size_t j, T val,
			    BMatrix<T>& obj, bool write){
  if(i >= this->ncols || j >= obj.GetNCols() ||
     this->nrows != obj.GetNRows()) return;
  sblcols->AddMultCol(SubBlockColFromIndex(i, false),
		      obj.SubBlockColFromIndex(j, false), val, *obj.sblcols);
  if(write) WriteSubBlockCols();
}

template <typename T>
BMatrix<T>* BMatrix<T>::GetBlock(size_t i, size_t m, size_t j, size_t n){
  string name = this->mname + "_bl_" + to_string(i) + "_" + to_string(m);
  name += "_" + to_string(j) + "_" + to_string(n);
  BMatrix<T>* out = new BMatrix(name, m, n,
				std::min(m/2, m*n/this->nrows),
				std::min(n/2, m*n/this->ncols), false,
				workdir+name+".h5");
  out->SetWorkDir(workdir);
  const size_t frbl = i / nblrows;
  const size_t lrbl = (i + m - 1) / nblrows;
  const size_t fcbl = j / nblcols;
  const size_t lcbl = (j + n - 1) / nblcols;
  const size_t frin = i % nblrows;
  const size_t fcin = j % nblcols;
  for(size_t irbl=frbl; irbl<=lrbl; irbl++)
    for(size_t icbl=fcbl; icbl<=lcbl; icbl++){
      ReadBlock(irbl, icbl);
      const size_t forbl = (irbl - frbl) * nblrows / out->nblrows;
      const size_t lorbl = (irbl - frbl + 1) * nblrows / out->nblrows;
      const size_t focbl = (icbl - fcbl) * nblcols / out->nblcols;
      const size_t locbl = (icbl - fcbl + 1) * nblcols / out->nblcols;
      for(size_t jrbl=forbl; jrbl<lorbl; jrbl++)
	for(size_t jcbl=focbl; jcbl<locbl; jcbl++){
	  size_t m = ((irbl == frbl) ? frin : 0) + (jrbl-forbl)*out->nblrows;
	  for(size_t i=0; i<out->nblrows; (i++, m++)){
	    size_t n = ((icbl == fcbl) ? fcin : 0) + (jcbl-focbl)*out->nblcols;
	    for(size_t j=0; j<out->nblcols; (j++, n++))
	      out->block->SetVal(i, j, block->GetVal(m, n));
	  }
	  out->SetBlockIndex(jrbl, jcbl);
	  out->WriteBlock();
	}
    }
  return out;
}

template <typename T>
PMatrix<T>* BMatrix<T>::GetPBlock(size_t i, size_t m, size_t j, size_t n){
  string name = this->mname + "_bl_" + to_string(i) + "_" + to_string(m);
  name += "_" + to_string(j) + "_" + to_string(n);
  PMatrix<T>* out = new PMatrix<T>(name, m, n, false);
  const size_t frbl = i / nblrows;
  const size_t lrbl = (i + out->GetNRows() - 1) / nblrows;
  const size_t fcbl = j / nblcols;
  const size_t lcbl = (j + out->GetNCols() - 1) / nblcols;
  const size_t frin = i % nblrows;
  const size_t fcin = j % nblcols;
  for(size_t irbl=frbl; irbl<=lrbl; irbl++)
    for(size_t icbl=fcbl; icbl<=lcbl; icbl++){
      ReadBlock(irbl, icbl);
      const size_t forbl = (irbl - frbl) * nblrows;
      const size_t lorbl = (irbl - frbl + 1) * nblrows;
      const size_t focbl = (icbl - fcbl) * nblcols;
      const size_t locbl = (icbl - fcbl + 1) * nblcols;
      size_t m = (irbl == frbl) ? frin : 0;
      for(size_t i=forbl; i<lorbl; (i++, m++)){
	size_t n = (icbl == fcbl) ? fcin : 0;
	for(size_t j=focbl; j<locbl; (j++, n++))
	  out->SetVal(i, j, block->GetVal(m, n));
      }
    }
  return out;
}

template <typename T>
void BMatrix<T>::SetBlock(size_t i, size_t j, BMatrix<T>& obj){
  const size_t frbl = i / nblrows;
  const size_t lrbl = (i + obj.nrows - 1) / nblrows;
  const size_t fcbl = j / nblcols;
  const size_t lcbl = (j + obj.ncols - 1) / nblcols;
  const size_t frin = i % nblrows;
  const size_t fcin = j % nblcols;
  for(size_t irbl=frbl; irbl<=lrbl; irbl++)
    for(size_t icbl=fcbl; icbl<=lcbl; icbl++){
      ReadBlock(irbl, icbl);
      const size_t forbl = (irbl - frbl) * nblrows / obj.nblrows;
      const size_t lorbl = (irbl - frbl + 1) * nblrows / obj.nblrows;
      const size_t focbl = (icbl - fcbl) * nblcols / obj.nblcols;
      const size_t locbl = (icbl - fcbl + 1) * nblcols / obj.nblcols;
      for(size_t jrbl=forbl; jrbl<lorbl; jrbl++)
	for(size_t jcbl=focbl; jcbl<locbl; jcbl++){
	  obj.ReadBlock(jrbl, jcbl);
	  size_t m = ((irbl == frbl) ? frin : 0) + (jrbl-forbl)*obj.nblrows;
	  for(size_t i=0; i<obj.nblrows; (i++, m++)){
	    size_t n = ((icbl == fcbl) ? fcin : 0) + (jcbl-focbl)*obj.nblcols;
	    for(size_t j=0; j<obj.nblcols; (j++, n++))
	      block->SetVal(m, n, obj.block->GetVal(i, j));
	  }
	}
      WriteBlock();
    }
}

template <typename T>
void BMatrix<T>::SetBlock(size_t i, size_t j, PMatrix<T> const& obj){
  const size_t frbl = i / nblrows;
  const size_t lrbl = (i + obj.GetNRows() - 1) / nblrows;
  const size_t fcbl = j / nblcols;
  const size_t lcbl = (j + obj.GetNCols() - 1) / nblcols;
  const size_t frin = i % nblrows;
  const size_t fcin = j % nblcols;
  for(size_t irbl=frbl; irbl<=lrbl; irbl++)
    for(size_t icbl=fcbl; icbl<=lcbl; icbl++){
      ReadBlock(irbl, icbl);
      const size_t forbl = (irbl - frbl) * nblrows;
      const size_t lorbl = (irbl - frbl + 1) * nblrows;
      const size_t focbl = (icbl - fcbl) * nblcols;
      const size_t locbl = (icbl - fcbl + 1) * nblcols;
      size_t m = (irbl == frbl) ? frin : 0;
      for(size_t i=forbl; i<lorbl; (i++, m++)){
	size_t n = (icbl == fcbl) ? fcin : 0;
	for(size_t j=focbl; j<locbl; (j++, n++))
	  block->SetVal(m, n, obj.GetVal(i, j));
      }
      WriteBlock();
    }
}

template <typename T> void BMatrix<T>::Random(){
  for(size_t icrb=0; icrb<blrcount; icrb++){
    SetBlockIndex(icrb, -1);
    for(currsblock=0; currsblock<sblrcount; currsblock++){
      sblrows->Random();
      WriteSubBlockRows();
    }
  }
}

template <typename T> void BMatrix<T>::BlPrint(size_t max_size, size_t prec,
					       size_t rbl,      size_t cbl){
  ReadBlock(rbl, cbl);
  block->Print(max_size, prec);
}

template <typename T> void BMatrix<T>::Print(){
  ReadBlock(0, 0);
  block->Print();
}

template <typename T> size_t BMatrix<T>::CurBlockRowOffset(){
  return currblock * nblrows;
}

template <typename T> size_t BMatrix<T>::CurBlockColOffset(){
  return curcblock * nblcols;
}

template <typename T> size_t BMatrix<T>::CurBlockNRows(){
  size_t offset = CurBlockRowOffset();
  if(offset >= this->nrows) return 0;
  return std::min(block->GetNRows(), this->nrows-offset);
}

template <typename T> size_t BMatrix<T>::CurBlockNCols(){
  size_t offset = CurBlockColOffset();
  if(offset >= this->ncols) return 0;
  return std::min(block->GetNCols(), this->ncols-offset);
}

template <typename T> size_t BMatrix<T>::CurSubBlockRowOffset(){
  return CurBlockRowOffset() + currsblock * sblrows->GetNRows();
}

template <typename T> size_t BMatrix<T>::CurSubBlockColOffset(){
  return CurBlockColOffset() + curcsblock * sblcols->GetNCols();
}

template <typename T> size_t BMatrix<T>::CurSubBlockNRows(){
  size_t offset = CurSubBlockRowOffset();
  if(offset >= this->nrows) return 0;
  return std::min(sblrows->GetNRows(), this->nrows-offset);
}

template <typename T> size_t BMatrix<T>::CurSubBlockNCols(){
  size_t offset = CurSubBlockColOffset();
  if(offset >= this->ncols) return 0;
  return std::min(sblcols->GetNCols(), this->ncols-offset);
}

template <typename T>
pair<size_t, size_t> BMatrix<T>::BlockIndexFromIndex(size_t i, size_t j,
						     bool breset){
  ReadBlock(i / nblrows, j / nblcols, breset);
  return make_pair(i % nblrows, j % nblcols);
}

template <typename T>
size_t BMatrix<T>::SubBlockRowFromIndex(size_t i, bool breset){
  currblock =  i / nblrows;
  ReadSubBlockRows((i % nblrows) / sblrows->GetNRows(), breset);
  return (i % nblrows) % sblrows->GetNRows();
}

template <typename T>
size_t BMatrix<T>::SubBlockColFromIndex(size_t i, bool breset){
  curcblock =  i / nblcols;
  ReadSubBlockCols((i % nblcols) / sblcols->GetNCols(), breset);
  return (i % nblcols) % sblcols->GetNCols();
}

template <typename T> void BMatrix<T>::SetBlockIndex(size_t i, size_t j){
  if(i == currblock && j == curcblock) return;
  currblock = i;
  curcblock = j;
  currsblock = -1;
  curcsblock = -1;
}

template <typename T>
void BMatrix<T>::SetCurrBlockNoRead(size_t i, size_t j){
  if(i >= blrcount || j >= blccount) return;
  if(i == currblock && j == curcblock) return;
  currblock = i;
  curcblock = j;
  block->ResetBuffer(true);
  currsblock = -1;
  curcsblock = -1;
}

template <typename T>
void BMatrix<T>::SetCurrBlockVal(size_t i, size_t j, T val){
  block->SetVal(i, j, val);
}

template <typename T>
T BMatrix<T>::GetCurrBlockVal(size_t i, size_t j){
  return block->GetVal(i, j);
}

template <typename T>
void BMatrix<T>::ReadBlock(size_t i, size_t j, bool breset){
  if(i >= blrcount || j >= blccount) return;
  if(i == currblock && j == curcblock) return;
  currblock = i;
  curcblock = j;
  hsize_t fstart[2] = {CurBlockRowOffset(), CurBlockColOffset()};
  hsize_t fblock[2] = {CurBlockNRows(),     CurBlockNCols()};
  filespace.selectHyperslab(H5S_SELECT_SET, fblock, fstart);
  hsize_t mstart[1] = {0};
  hsize_t mblock[1] = {fblock[0]*fblock[1]};
  DataSpace memspace(1, mblock);
  memspace.selectHyperslab(H5S_SELECT_SET, mblock, mstart);
  dataset->read(&block->GetData().front(),
		datatype, memspace, filespace);
  block->ResetBuffer(breset);
  memspace.selectNone();
  filespace.selectNone();
  currsblock = -1;
  curcsblock = -1;
}

template <typename T> void BMatrix<T>::ReadSubBlockRows(size_t i, bool breset){
  if(i >= sblrcount)  return;
  if(i == currsblock) return;
  currsblock = i;
  hsize_t fstart[2] = {CurSubBlockRowOffset(), 0};
  hsize_t fblock[2] = {CurSubBlockNRows(), sblrows->GetNCols()};
  filespace.selectHyperslab(H5S_SELECT_SET, fblock, fstart);
  hsize_t mstart[1] = {0};
  hsize_t mblock[1] = {fblock[0]*fblock[1]};
  DataSpace memspace(1, mblock);
  memspace.selectHyperslab(H5S_SELECT_SET, mblock, mstart);
  dataset->read(&sblrows->GetData().front(),
		datatype, memspace, filespace);
  sblrows->ResetBuffer(breset);
  memspace.selectNone();
  filespace.selectNone();
}

template <typename T> void BMatrix<T>::ReadSubBlockCols(size_t i, bool breset){
  if(i >= sblccount)  return;
  if(i == curcsblock) return;
  curcsblock = i;
  hsize_t fstart[2] = {0, CurSubBlockColOffset()};
  hsize_t fblock[2] = {sblcols->GetNRows(), CurSubBlockNCols()};
  filespace.selectHyperslab(H5S_SELECT_SET, fblock, fstart);
  hsize_t mstart[1] = {0};
  hsize_t mblock[1] = {fblock[0]*fblock[1]};
  DataSpace memspace(1, mblock);
  memspace.selectHyperslab(H5S_SELECT_SET, mblock, mstart);
  dataset->read(&sblcols->GetData().front(),
		datatype, memspace, filespace);
  sblcols->ResetBuffer(breset);
  memspace.selectNone();
  filespace.selectNone();
}

template <typename T> void BMatrix<T>::WriteBlock(){
  if(currblock >= blrcount || curcblock >= blccount) return;
  hsize_t fstart[2] = {CurBlockRowOffset(), CurBlockColOffset()};
  hsize_t fblock[2] = {CurBlockNRows(),     CurBlockNCols()};
  filespace.selectHyperslab(H5S_SELECT_SET, fblock, fstart);
  hsize_t mstart[1] = {0};
  hsize_t mblock[1] = {fblock[0]*fblock[1]};
  DataSpace memspace(1, mblock);
  memspace.selectHyperslab(H5S_SELECT_SET, mblock, mstart);
  dataset->write(&block->GetData().front(),
		 datatype, memspace, filespace);
  memspace.selectNone();
  filespace.selectNone();
}

template <typename T> void BMatrix<T>::WriteSubBlockRows(){
  if(currsblock >= sblrcount) return;
  hsize_t fstart[2] = {CurSubBlockRowOffset(), 0};
  hsize_t fblock[2] = {CurSubBlockNRows(), sblrows->GetNCols()};
  filespace.selectHyperslab(H5S_SELECT_SET, fblock, fstart);
  hsize_t mstart[1] = {0};
  hsize_t mblock[1] = {fblock[0]*fblock[1]};
  DataSpace memspace(1, mblock);
  memspace.selectHyperslab(H5S_SELECT_SET, mblock, mstart);
  dataset->write(&sblrows->GetData().front(),
		 datatype, memspace, filespace);
  memspace.selectNone();
  filespace.selectNone();
}

template <typename T> void BMatrix<T>::WriteSubBlockCols(){
  if(curcsblock >= sblccount) return;
  hsize_t fstart[2] = {0, CurSubBlockColOffset()};
  hsize_t fblock[2] = {sblcols->GetNRows(), CurSubBlockNCols()};
  filespace.selectHyperslab(H5S_SELECT_SET, fblock, fstart);
  hsize_t mstart[1] = {0};
  hsize_t mblock[1] = {fblock[0]*fblock[1]};
  DataSpace memspace(1, mblock);
  memspace.selectHyperslab(H5S_SELECT_SET, mblock, mstart);
  dataset->write(&sblcols->GetData().front(),
		 datatype, memspace, filespace);
  memspace.selectNone();
  filespace.selectNone();
}

#endif

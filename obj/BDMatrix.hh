#ifndef __BDMATRIX__
#define __BDMATRIX__

#include "PMatrix.hh"
#include "HDF5Util.hh"
#include <stdlib.h>
#include <string>
#include <cstring>
#include <regex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <utility>
#include <assert.h>
#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <H5Cpp.h>
#include <cblas.h>
#include <lapack.h>
#include <lapacke.h>

using namespace std;

template <typename T> class BDMatrix{
public:
  BDMatrix();
  BDMatrix(string name);
  BDMatrix(string name, BDMatrix<T> const& obj);
  BDMatrix(string name, PMatrix<T>& obj, T thresh, bool diagonalize=true);
  BDMatrix(string name, string base, H5::H5File const& infile,
	   bool blocks=true, bool diagonalize=true);
  BDMatrix(string name, string base,
	   TFile* const infile, bool diagonalize=true);
  BDMatrix(string name, string base, string infile,
	   ROOT::R::TRInterface& R, bool diagonalize=true);
  virtual ~BDMatrix();

  virtual void AppendBlock(PMatrix<T>* block, bool diagonalize=true);
  virtual void AppendHDF5Block(string name, H5::H5File const& infile);
  virtual void AppendROOTBlock(string name,
			       TFile* const infile, bool diagonalize=true);
  virtual void AppendRdataBlock(string name, string infile,
				ROOT::R::TRInterface& R,bool diagonalize=true);
  virtual void SortBlocks();
  virtual PMatrix<T>* GetBlock(size_t bindex) const;
  virtual void GetBlock(size_t bindex, PMatrix<T>& block) const;
  virtual void CopyBlock(size_t bindex, PMatrix<T>* block);
  virtual void ClearBlocks();

  bool operator==(BDMatrix<T> const& obj);
  bool operator&(BDMatrix<T>  const& obj) const;

  BDMatrix<T>* operator+(BDMatrix<T> const& obj);
  BDMatrix<T>* operator-(BDMatrix<T> const& obj);
  BDMatrix<T>* operator*(BDMatrix<T> const& obj);
  BDMatrix<T>* operator*(T val);

  BDMatrix<T>& operator+=(BDMatrix<T> const& obj);
  BDMatrix<T>& operator-=(BDMatrix<T> const& obj);
  BDMatrix<T>& operator*=(T val);

  virtual BDMatrix<T>* axpy_oop(string name, T alpha, BDMatrix<T> const& obj);
  virtual void axpy(T alpha, BDMatrix<T> const& obj);
  virtual void RightMult(BDMatrix<T> const& obj, BDMatrix<T>* out) const;

  virtual string GetName()  const { return mname; }
  virtual size_t GetNRows() const { return nrows; }
  virtual size_t GetNCols() const { return nrows; }
  virtual size_t GetNData() const { return data.size(); }
  virtual vector<size_t> GetIndexPermutation() const { return permutation;  }
  virtual vector<size_t> GetBlockPermutation() const { return bpermutation; }
  virtual size_t GetIndexPermutation(size_t index)  const;
  virtual size_t GetBlockPermutation(size_t bindex) const;
  virtual size_t GetBlockOffset(size_t bindex) const;
  virtual size_t GetBlockDataOffset(size_t bindex) const;
  
  virtual T GetVal(size_t index) const;
  virtual T GetVal(size_t row, size_t col) const;
  virtual T GetVal(size_t bindex, size_t brow, size_t bcol) const;
  virtual vector<T>& GetData(){ return data; }
  virtual const vector<T>& GetDataConst() const{ return data; }
  virtual void SetVal(size_t row, size_t col, T val);
  virtual void SetVal(size_t bindex, size_t brow, size_t bcol, T val);
  virtual void AddVal(size_t row, size_t col, T val);
  virtual void AddVal(size_t bindex, size_t brow, size_t bcol, T val);
  virtual void MultVal(size_t row, size_t col, T val);
  virtual void MultVal(size_t bindex, size_t brow, size_t bcol, T val);  
  virtual void SetData(const vector<T>& d);
  virtual pair<size_t, size_t> IndexToBlockIndex(const size_t& ind) const;
  
  virtual size_t GetNBlocks() const { return bdim.size(); }
  virtual size_t GetBlockRows(size_t bindex)  const;
  virtual const vector<size_t>& GetBlockRows() const { return bdim; }
  virtual size_t GetBlockCols(size_t bindex)  const;
  virtual const vector<size_t>& GetBlockCols() const { return bdim; }
  virtual size_t GetBlockNData(size_t bindex) const;
  virtual const vector<size_t>& GetBlockNData() const { return bsize; }
  virtual T GetBlockVal(size_t bindex, size_t index) const;
  virtual T GetBlockVal(size_t bindex, size_t row, size_t col) const;
  virtual size_t BlockIndexToIndex(size_t bindex, size_t index) const;

  virtual T GetThreshold() const { return threshold; }
  virtual void SetThreshold(T thresh){ threshold = thresh; }

  virtual bool GetUseCblas() const { return use_cblas; }
  virtual void SetUseCblas(bool cb){ use_cblas = cb; }

  virtual void SetAll(T val);
  virtual void Zero();
  virtual void ZeroBlock(size_t bindex);

  virtual BDMatrix<T>* Inverse(double* logdet=NULL) const;
  virtual void Invert(double* logdet=NULL);
  virtual BDMatrix<T>* BlockwiseInverse(size_t minbl=16,
					double* logdet=NULL) const;
  virtual void BlockwiseInvert(size_t minbl=16, double* logdet=NULL);
  virtual BDMatrix<T>* Sqrt();
  virtual T Trace();
  virtual T TraceProduct(BDMatrix<T>& obj);
  virtual T TraceProductSymm(BDMatrix<T>& obj);
  virtual void Transpose();

  virtual void Random();

  virtual void RightMult(PMatrix<T>& obj, PMatrix<T>* out) const;

  virtual void Print(size_t max_bl, size_t max_size,
		     size_t prec, string lead="") const;
  virtual void Print(size_t max_bl=0) const;

  virtual void WriteHDF51d(H5::H5File& outfile, string name="",
			   size_t chunk=0, int compression=0) const;
  virtual void WriteHDF5(H5::H5File &outfile, string name="") const;
  virtual void WriteROOT(TFile* outfile,      string name="") const;
  virtual void WriteRdata(string outfile,     string name,
			  ROOT::R::TRInterface& R);
  virtual vector<string> AddToREnv(string name, ROOT::R::TRInterface& R);

protected:
  string mname;
  size_t nrows;
  vector<size_t> bdim;
  vector<size_t> cdim;
  vector<size_t> bsize;
  vector<size_t> pblock;
  vector<size_t> permutation;
  vector<size_t> bpermutation;
  vector<T> data;
  T defval;
  T threshold;
  bool use_cblas;
};

template <typename T> BDMatrix<T>::BDMatrix(){
  mname = "";
  nrows = 0;
  cdim.push_back(0);
  pblock.push_back(0);
  defval = T();
  threshold = T(1.0e-12);
  use_cblas = true;
}

template <typename T> BDMatrix<T>::BDMatrix(string name) : BDMatrix(){
  mname = name;
}

template <typename T>
BDMatrix<T>::BDMatrix(string name, BDMatrix<T> const& obj) :
  BDMatrix(name){
  nrows        = obj.nrows;
  bdim         = obj.bdim;
  cdim         = obj.cdim;
  bsize        = obj.bsize;
  pblock       = obj.pblock;
  permutation  = obj.permutation;
  bpermutation = obj.bpermutation;
  data         = obj.data;
  threshold    = obj.threshold;
  use_cblas    = obj.use_cblas;
}

template <typename T>
BDMatrix<T>::BDMatrix(string name,
		      PMatrix<T>& obj, T thresh, bool diagonalize) :
  BDMatrix(name){
  threshold = thresh;
  vector<set<size_t> > blocks;
  if(diagonalize) blocks = obj.BlockDiagonalize(threshold);
  else{
    vector<size_t> tmp(obj.GetNRows());
    iota(tmp.begin(), tmp.end(), 1);
    blocks.push_back(set<size_t>(blocks.begin(), blocks.end()));
  }
  PMatrix<double>* bl = NULL;
  vector<size_t> tmpperm;
  for(size_t i=0; i<blocks.size(); i++){
    const size_t n = blocks[i].size();
    if(bl)
      if(bl->GetNRows() != n){
	delete bl;
	bl = NULL;
      }
    if(!bl) bl = new PMatrix<double>("tmp", n, n, false);
    vector<size_t> vbl(blocks[i].begin(), blocks[i].end());
    for(size_t j=0; j<n; j++){
      for(size_t k=0; k<n; k++)
	bl->SetVal(j, k, obj.GetVal(vbl[j], vbl[k]));
      tmpperm.push_back(vbl[j]);
    }
    AppendBlock(bl, false);
  }
  if(bl) delete bl;
  copy(tmpperm.begin(), tmpperm.end(), permutation.begin());
}

template <typename T>
BDMatrix<T>::BDMatrix(string name, string base,
		      H5::H5File const& infile, bool blocks,bool diagonalize) :
  BDMatrix(name){
  size_t iblock = 0;
  if(blocks){
    while(H5Lexists(infile.getId(),
		    (base+to_string(iblock)).c_str(), H5P_DEFAULT)){
      AppendHDF5Block(base+to_string(iblock), infile);
      iblock ++;
    }
  }
  else{
    vector<size_t> pbl;
    vector<T> dtmp;
    ReadHDF5vector<size_t>(&infile, pbl, base+"_dim");
    ReadHDF5vector<T>(&infile, dtmp, base+"_data");
    PMatrix<T>* bl = new PMatrix<T>();
    for(size_t i=0; i<pbl.size()-1; i++){
      const size_t n = pbl[i+1] - pbl[i];
      const double m = sqrt(n);
      assert((double) ((size_t) m) == m);
      if(m != bl->GetNRows()){
	delete bl;
	bl = new PMatrix<T>("sbl", (size_t) m, (size_t) m, false);
      }
      copy(dtmp.begin()+pbl[i], dtmp.begin()+pbl[i]+n, bl->GetData().begin());
      AppendBlock(bl, diagonalize);
    }
    delete bl;
  }
}

template <typename T>
BDMatrix<T>::BDMatrix(string name, string base,
		      TFile* const infile, bool diagonalize) :
  BDMatrix(name){
  size_t iblock = 0;
  TList* keys = infile->GetListOfKeys();
  while(keys->Contains((base+to_string(iblock)).c_str())){
    AppendROOTBlock(base+to_string(iblock), infile, diagonalize);
    iblock ++;
  }
}

template <typename T>
BDMatrix<T>::BDMatrix(string name, string base, string infile,
		      ROOT::R::TRInterface& R, bool diagonalize) :
  BDMatrix(name){
  size_t iblock = 0;
  int nkeys = 0;
  vector<string> keys;
  R << ("load(file.path(\"" + infile + "\"), dfenv <- new.env())").c_str();
  R << "keys <- dfenv.ls()";
  R << "nkeys <- count(keys)";
  R["nkeys"] >> nkeys;
  keys.resize(nkeys);
  R["keys"] >> keys;
  while(find(keys.begin(), keys.end(), base+to_string(iblock)) != keys.end()){
    AppendRdataBlock(base+to_string(iblock), infile, diagonalize);
    iblock ++;
  }
}

template <typename T> BDMatrix<T>::~BDMatrix(){
}

template <typename T> void BDMatrix<T>::AppendBlock(PMatrix<T>* block,
						    bool diagonalize){
  if(block->GetNRows() != block->GetNCols()){
    cout << "BDMatrix::AddBlockAtIndex - block is not square" << endl;
    return;
  }
  vector<set<size_t> > sub_blocks;
  if(diagonalize) sub_blocks = block->BlockDiagonalize(threshold);
  else{
    vector<size_t> tmp(block->GetNRows());
    iota(tmp.begin(), tmp.end(), 0);
    sub_blocks.push_back(set<size_t>(tmp.begin(), tmp.end()));
  }
  size_t index = cdim[cdim.size()-1];
  size_t offset = 0;
  for(const auto& s : sub_blocks){
    nrows += s.size();
    bdim.push_back(s.size());
    cdim.push_back(cdim.back() + s.size());
    bsize.push_back(s.size() * s.size());
    pblock.push_back(pblock.back() + bsize.back());
    for(auto it=s.begin(); it!=s.end(); it++)
      permutation.push_back(*it + index);
    bpermutation.push_back(bpermutation.size());
    PMatrix<T>* sbl = new PMatrix<T>("sbl", s.size(), s.size(), false);
    for(size_t i=0; i<s.size(); i++)
      for(size_t j=0; j<s.size(); j++)
	sbl->SetVal(i, j, block->GetVal(offset+i, offset+j));
    offset += s.size();
    data.insert(data.end(),
		sbl->GetDataConst().begin(), sbl->GetDataConst().end());
    delete sbl;
  }
}

template <typename T>
void BDMatrix<T>::AppendHDF5Block(string name, H5::H5File const& infile){
  H5::DataSet   dset   = infile.openDataSet(name);
  H5::DataSpace dspace = dset.getSpace();
  H5T_class_t typeclass = dset.getTypeClass();
  size_t dsize = 0;
  const size_t tmpsize = data.size();
  size_t rank = dspace.getSimpleExtentNdims();
  assert(rank == 1 || rank == 2);
  hsize_t dims[2] = {1, 1};
  dspace.getSimpleExtentDims(dims, NULL);
  assert(dims[0] > 0 && dims[1] > 0 && dims[0] == dims[1]);
  if(typeclass == H5T_INTEGER){
    H5::IntType inttype = dset.getIntType();
    dsize = inttype.getSize();
    if(is_same<T, int>::value){
      if(sizeof(int) != dsize){
	cout << "BDMatrix of type int, but HDF5 int size is " << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(tmpsize + dims[0] * dims[1]);
	dset.read(&data.front()+tmpsize, H5::PredType::NATIVE_INT);
      }
    }
    else if(is_same<T, long>::value){
      if(sizeof(long) != dsize){
	cout << "BDMatrix of type long, but HDF5 int size is " << dsize <<endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(tmpsize + dims[0] * dims[1]);
	dset.read(&data.front()+tmpsize, H5::PredType::NATIVE_LONG);
      }
    }
    else{
      cout << "BDMatrix unsupported int type of size " << dsize << endl;
      dspace.close(); dset.close();
      return;
    }
  }
  else if(typeclass == H5T_FLOAT){
    H5::FloatType floattype = dset.getFloatType();
    dsize = floattype.getSize();
    if(is_same<T, float>::value){
      if(sizeof(float) != dsize){
	cout << "BDMatrix of type float, but HDF5 float size is "
	     << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(tmpsize + dims[0] * dims[1]);
	dset.read(&data.front()+tmpsize, H5::PredType::NATIVE_FLOAT);
      }
    }
    else if(is_same<T, double>::value){
      if(sizeof(double) != dsize){
	cout << "BDMatrix of type double, but HDF5 float size is "
	     << dsize << endl;
	dspace.close(); dset.close();
	return;
      }
      else{
	data.resize(tmpsize + dims[0] * dims[1]);
	dset.read(&data.front()+tmpsize, H5::PredType::NATIVE_DOUBLE);
      }
    }
    else{
      cout << "BDMatrix unsupported float type of size " << dsize << endl;
      dspace.close(); dset.close();
      return;
    }
  }
  else{
    cout << "BDMatrix unsupported HDF5 type class " << typeclass << endl;
    dspace.close(); dset.close();
    return;
  }
  dspace.close(); dset.close();
  nrows += dims[0];
  bdim.push_back(dims[0]);
  cdim.push_back(cdim.back() + dims[0]);
  bsize.push_back(dims[0] * dims[1]);
  pblock.push_back(pblock.back() + bsize.back());
  for(size_t i=tmpsize; i<data.size(); i++)
    permutation.push_back(i);
  bpermutation.push_back(bpermutation.size());
}

template <typename T>
void BDMatrix<T>::AppendROOTBlock(string name,
				  TFile* const infile, bool diagonalize){
  PMatrix<T>* tmp = new PMatrix<T>(name, infile);
  AppendBlock(tmp, diagonalize);
  delete tmp;
}

template <typename T>
void BDMatrix<T>::AppendRdataBlock(string name, string infile,
				   ROOT::R::TRInterface& R, bool diagonalize){
  PMatrix<T>* tmp = new PMatrix<T>(name, infile, R);
  AppendBlock(tmp, diagonalize);
  delete tmp;
}

template <typename T> void BDMatrix<T>::SortBlocks(){
  vector<size_t> p(bdim.size());
  iota(p.begin(), p.end(), 0);
  sort(p.begin(), p.end(), [this](const size_t i, const size_t j){
    if(bdim[i] != bdim[j]) return bdim[i] > bdim[j];
    else return i < j; });
  vector<size_t> btmp(bpermutation.begin(), bpermutation.end());
  fill(bpermutation.begin(), bpermutation.end(), T(0));
  transform(p.begin(), p.end(), bpermutation.begin(),
	    [this,&btmp](const size_t i){ return btmp[i]; });
  BDMatrix<T>* tmp = new BDMatrix<T>("tmp", *this);
  ClearBlocks();
  PMatrix<T>* ptmp = new PMatrix<T>();
  for(size_t i=0; i<tmp->GetNBlocks(); i++){
    size_t j = tmp->bpermutation[i];
    tmp->GetBlock(j, *ptmp);
    AppendBlock(ptmp);
    size_t offset = permutation.size() - ptmp->GetNRows();
    for(size_t k=tmp->cdim[j]; k<tmp->cdim[j+1]; (k++, offset++))
      permutation[offset] = tmp->permutation[k];
  }
  bpermutation = tmp->bpermutation;
  delete tmp;
  if(ptmp) delete ptmp;
}

template <typename T> PMatrix<T>* BDMatrix<T>::GetBlock(size_t bindex) const{
  PMatrix<T>* block = new PMatrix<T>();;
  GetBlock(bindex, *block);
  return block;
}

template <typename T>
void BDMatrix<T>::GetBlock(size_t bindex, PMatrix<T>& block) const{
  if(bindex >= bdim.size()) return;
  if(block.GetNRows() != bdim[bindex] || block.GetNCols() != bdim[bindex]){
    block = PMatrix<T>(mname+"_bl"+to_string(bindex),
		       bdim[bindex], bdim[bindex], false);
    block.SetUseCblas(use_cblas);
  }
  copy(data.begin()+pblock[bindex],
       data.begin()+pblock[bindex]+bsize[bindex], block.GetData().begin());
  block.ResetBuffer();
}

template <typename T>
void BDMatrix<T>::CopyBlock(size_t bindex, PMatrix<T>* block){
  if(bindex >= bdim.size()) return;
  if(block->GetNRows()!=bdim[bindex] || block->GetNCols()!=bdim[bindex]) return;
  copy(block->GetDataConst().begin(),
       block->GetDataConst().end(), data.begin() + pblock[bindex]);
}

template <typename T> void BDMatrix<T>::ClearBlocks(){
  bdim.clear();
  cdim.clear();
  bsize.clear();
  pblock.clear();
  permutation.clear();
  bpermutation.clear();
  data.clear();
  nrows = 0;
  cdim.push_back(0);
  pblock.push_back(0);
}

template <typename T> bool BDMatrix<T>::operator==(BDMatrix<T> const& obj){
  if(!(*this & obj)) return false;
  for(size_t i=0; i<data.size(); i++)
    if(abs(data[i]-obj.data[i]) >= threshold) return false;
  return true;
}

template <typename T> bool BDMatrix<T>::operator&(BDMatrix<T> const& obj) const{
  return bdim == obj.bdim;
}

template <typename T>
BDMatrix<T>* BDMatrix<T>::operator+(BDMatrix<T> const& obj){
  if(*this & obj) return axpy_oop(mname+"_"+obj.GetName()+"_sum", T(1), obj);
  else return new BDMatrix<T>();
}

template <typename T>
BDMatrix<T>* BDMatrix<T>::operator-(BDMatrix<T> const& obj){
  if(*this & obj) return axpy_oop(mname+"_"+obj.GetName()+"_diff", T(-1), obj);
  else return new BDMatrix<T>();
}

template <typename T>
BDMatrix<T>* BDMatrix<T>::operator*(BDMatrix<T> const& obj){
  if(*this & obj){
    BDMatrix<T>* out = new BDMatrix<T>(mname+"_"+obj.GetName()+"_mult", *this);
    RightMult(obj, out);
    return out;
  }
  else return new BDMatrix<T>();
}

template <typename T> BDMatrix<T>* BDMatrix<T>::operator*(T val){
  BDMatrix<T>* out = new BDMatrix<T>(mname+"_scal", *this);
  *out *= val;
  return out;
}

template <typename T>
BDMatrix<T>& BDMatrix<T>::operator+=(BDMatrix<T> const& obj){
  axpy(T(1), obj);
  return *this;
}

template <typename T>
BDMatrix<T>& BDMatrix<T>::operator-=(BDMatrix<T> const& obj){
  axpy(T(-1), obj);
  return *this;
}

template <typename T> BDMatrix<T>& BDMatrix<T>::operator*=(T val){
  if(val == T(1)) return *this;
  if(use_cblas && is_same<T, float>::value)
    cblas_sscal(data.size(), (float) val, (float*) &data.front(), 1);
  else if(use_cblas && is_same<T, double>::value)
    cblas_dscal(data.size(), (double) val, (double*) &data.front(), 1);
  else
    for(size_t i=0; i<data.size(); i++) data[i] *= val;
  return *this;
}

template <typename T>
BDMatrix<T>* BDMatrix<T>::axpy_oop(string name, T alpha,BDMatrix<T> const& obj){
  if(!(*this & obj)) return new BDMatrix<T>();
  BDMatrix<T>* out = new BDMatrix(name, *this);
  if(use_cblas && is_same<T, float>::value && data.size() <= INT_MAX)
    cblas_saxpy(out->data.size(), (float) alpha,
		(float*) &obj.data.front(), 1, (float*) &out->data.front(), 1);
  else if(use_cblas && is_same<T, double>::value && data.size() <= INT_MAX)
    cblas_daxpy(out->data.size(), (double) alpha,
		(double*)&obj.data.front(), 1, (double*)&out->data.front(), 1);
  else if(alpha == T(1))
    for(size_t i=0; i<out->data.size(); i++) out->data[i] += obj.data[i];
  else if(alpha == T(-1))
    for(size_t i=0; i<out->data.size(); i++) out->data[i] -= obj.data[i];
  else
    for(size_t i=0; i<out->data.size(); i++) out->data[i] += alpha*obj.data[i];
  return out;
}

template <typename T> void BDMatrix<T>::axpy(T alpha, BDMatrix<T> const& obj){
  if(!(*this & obj)) return;
  if(use_cblas && is_same<T, float>::value && data.size() <= INT_MAX)
    cblas_saxpy(data.size(), (float) alpha,
		(float*) &obj.data.front(), 1, (float*) &data.front(), 1);
  else if(use_cblas && is_same<T, float>::value && data.size() <= INT_MAX)
    cblas_daxpy(data.size(), (double) alpha,
		(double*) &obj.data.front(), 1, (double*) &data.front(), 1);
  else if(alpha == T(1))
    for(size_t i=0; i<data.size(); i++) data[i] += obj.data[i];
  else if(alpha == T(-1))
    for(size_t i=0; i<data.size(); i++) data[i] -= obj.data[i];
  else
    for(size_t i=0; i<data.size(); i++) data[i] += alpha * obj.data[i];
}

template <typename T>
void BDMatrix<T>::RightMult(BDMatrix<T> const& obj, BDMatrix<T>* out) const{
  if(!(*this & obj)) return;
  if(!out) out = new BDMatrix<T>(mname+"_mult_"+obj.mname, *this);
  out->Zero();
  PMatrix<T>* bl    = new PMatrix<T>();
  PMatrix<T>* blobj = new PMatrix<T>();
  PMatrix<T>* blout = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    obj.GetBlock(i, *blobj);
    out->GetBlock(i, *blout);
    if(bl && blobj && blout){
      bl->RightMult(*blobj, blout);
      out->CopyBlock(i, blout);
    }
  }
  if(bl)    delete bl;
  if(blobj) delete blobj;
  if(blout) delete blout;
}

template <typename T>
size_t BDMatrix<T>::GetIndexPermutation(size_t index) const{
  return index < permutation.size() ? permutation[index] : 0;
}

template <typename T>
size_t BDMatrix<T>::GetBlockPermutation(size_t bindex) const{
  return bindex < bpermutation.size() ? bpermutation[bindex] : 0;
}

template <typename T>
size_t BDMatrix<T>::GetBlockOffset(size_t bindex) const{
  return bindex < bpermutation.size() ? cdim[bindex] : 0;
}

template <typename T>
size_t BDMatrix<T>::GetBlockDataOffset(size_t bindex) const{
  return bindex < bpermutation.size() ? pblock[bindex] : 0;
}

template <typename T> T BDMatrix<T>::GetVal(size_t index) const{
  return (index < data.size() ? data[index] : T(0));
}

template <typename T> T BDMatrix<T>::GetVal(size_t row, size_t col) const{
  if(row >= nrows || col >= nrows) return T(0);
  pair<size_t, size_t> rindex = IndexToBlockIndex(row);
  pair<size_t, size_t> cindex = IndexToBlockIndex(col);
  if(rindex.first != cindex.first || rindex.first  >= bdim.size()) return T(0);
  if(rindex.second >= bdim[rindex.first] ||
     cindex.second >= bdim[cindex.first]) return T(0);
  return data[pblock[rindex.first] +
	      rindex.second*bdim[rindex.first] + cindex.second];
}

template <typename T> T BDMatrix<T>::GetVal(size_t bindex,
					    size_t brow, size_t bcol) const{
  if(bindex > GetNBlocks()) return T(0);
  if(brow >= bdim[bindex] || bcol >= bdim[bindex]) return T(0);
  return data[pblock[bindex] + bcol + brow*bdim[bindex]];  
}

template <typename T> void BDMatrix<T>::SetVal(size_t row, size_t col, T val){
  if(row >= nrows || col >= nrows) return;
  pair<size_t, size_t> rindex = IndexToBlockIndex(row);
  pair<size_t, size_t> cindex = IndexToBlockIndex(col);
  if(rindex.first != cindex.first || rindex.first  >= bdim.size()) return;
  if(rindex.second >= bdim[rindex.first] ||
     cindex.second >= bdim[cindex.first]) return;
  data[pblock[rindex.first] +
       rindex.second*bdim[rindex.first] + cindex.second] = val;
}

template <typename T> void BDMatrix<T>::SetVal(size_t bindex,
					       size_t brow, size_t bcol,T val){
  if(bindex > GetNBlocks()) return;
  if(brow >= bdim[bindex] || bcol >= bdim[bindex]) return;
  data[pblock[bindex] + bcol + brow*bdim[bindex]] = val;
}

template <typename T> void BDMatrix<T>::AddVal(size_t row, size_t col, T val){
  if(row >= nrows || col >= nrows) return;
  pair<size_t, size_t> rindex = IndexToBlockIndex(row);
  pair<size_t, size_t> cindex = IndexToBlockIndex(col);
  if(rindex.first != cindex.first || rindex.first  >= bdim.size()) return;
  if(rindex.second >= bdim[rindex.first] ||
     cindex.second >= bdim[cindex.first]) return;
  data[pblock[rindex.first] +
       rindex.second*bdim[rindex.first] + cindex.second] += val;
}

template <typename T> void BDMatrix<T>::AddVal(size_t bindex,
					       size_t brow, size_t bcol,T val){
  if(bindex > GetNBlocks()) return;
  if(brow >= bdim[bindex] || bcol >= bdim[bindex]) return;
  data[pblock[bindex] + bcol + brow*bdim[bindex]] += val;
}

template <typename T> void BDMatrix<T>::MultVal(size_t row, size_t col, T val){
  if(row >= nrows || col >= nrows) return;
  pair<size_t, size_t> rindex = IndexToBlockIndex(row);
  pair<size_t, size_t> cindex = IndexToBlockIndex(col);
  if(rindex.first != cindex.first || rindex.first  >= bdim.size()) return;
  if(rindex.second >= bdim[rindex.first] ||
     cindex.second >= bdim[cindex.first]) return;
  data[pblock[rindex.first] +
       rindex.second*bdim[rindex.first] + cindex.second] *= val;
}

template <typename T> void BDMatrix<T>::MultVal(size_t bindex,
						size_t brow,size_t bcol,T val){
  if(bindex > GetNBlocks()) return;
  if(brow >= bdim[bindex] || bcol >= bdim[bindex]) return;
  data[pblock[bindex] + bcol + brow*bdim[bindex]] *= val;
}

template <typename T> void BDMatrix<T>::SetData(const vector<T>& d){
  assert(d.size() == data.size());
  copy(d.begin(), d.end(), data.begin());
}

template <typename T>
pair<size_t, size_t> BDMatrix<T>::IndexToBlockIndex(const size_t& ind) const{
  pair<size_t, size_t> pr(-1, -1);
  if(ind >= nrows) return pr;
  auto it = lower_bound(cdim.begin(), cdim.end(), ind, less_equal<int>());
  if(it == cdim.end()) return pr;
  pr.first = distance(cdim.begin(), it) - 1;
  pr.second = ind - cdim[pr.first];
  return pr;
}
  
template <typename T> size_t BDMatrix<T>::GetBlockRows(size_t bindex) const{
  return (bindex < bdim.size() ? bdim[bindex] : 0);
}

template <typename T> size_t BDMatrix<T>::GetBlockCols(size_t bindex) const{
  return GetBlockRows(bindex);
}

template <typename T> size_t BDMatrix<T>::GetBlockNData(size_t bindex) const{
  return (bindex < bdim.size() ? bsize[bindex] : 0);
}

template <typename T>
T BDMatrix<T>::GetBlockVal(size_t bindex, size_t index) const{
  if(bindex >= bdim.size()) return T(0);
  if(index >= bsize[bindex]) return T(0);
  return data[pblock[bindex]+index];
}

template <typename T>
T BDMatrix<T>::GetBlockVal(size_t bindex, size_t row, size_t col) const{
  if(bindex >= bdim.size()) return T(0);
  if(row >= bdim[bindex] || col >= bdim[bindex]) return T(0);
  return data[pblock[bindex] + col+row*bdim[bindex]];
}

template <typename T>
size_t BDMatrix<T>::BlockIndexToIndex(size_t bindex, size_t index) const{
  if(bindex >= bdim.size()) return -1;
  if(index >= bsize[bindex]) return -1;
  return cdim[bindex] + index;
}

template <typename T> void BDMatrix<T>::SetAll(T val){
  fill(data.begin(), data.end(), val);
}

template <typename T> void BDMatrix<T>::Zero(){
  SetAll(T(0));
}

template <typename T> void BDMatrix<T>::ZeroBlock(size_t bindex){
  if(bindex >= bdim.size()) return;
  for(size_t i=0; i<bdim[bindex]; i++)
    for(size_t j=0; j<bdim[bindex]; j++)
      SetVal(bindex, i, j, 0.0);
}

template <typename T>
BDMatrix<T>* BDMatrix<T>::Inverse(double* logdet) const{
  BDMatrix<T>* out = new BDMatrix<T>(mname+"_inv", *this);
  out->Invert(logdet);
  return out;
}

template <typename T> void BDMatrix<T>::Invert(double* logdet){
  PMatrix<T>* bl = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    if(bl){
      bl->Invert(logdet);
      CopyBlock(i, bl);
    }
  }
  if(bl) delete bl;
}

template <typename T>
BDMatrix<T>* BDMatrix<T>::BlockwiseInverse(size_t minbl, double* logdet) const{
  BDMatrix<T>* out = new BDMatrix<T>(mname+"_inv", *this);
  out->BlockwiseInvert(minbl, logdet);
  return out;
}

template <typename T>
void BDMatrix<T>::BlockwiseInvert(size_t minbl, double* logdet){
  PMatrix<T>* bl = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    if(bl){
      bl->BlockwiseInvert(minbl, logdet);
      CopyBlock(i, bl);
    }
  }
  if(bl) delete bl;
}

template <typename T> BDMatrix<T>* BDMatrix<T>::Sqrt(){
  BDMatrix<T>* out = new BDMatrix<T>(mname+"_sqrt", *this);
  out->Zero();
  PMatrix<T>* bl = new PMatrix<T>();
  PMatrix<T>* sq = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    out->GetBlock(i, *sq);
    if(bl && sq){
      bl->Sqrt(sq);
      out->CopyBlock(i, sq);
    }
  }
  if(bl) delete bl;
  if(sq) delete sq;
  return out;
}

template <typename T> T BDMatrix<T>::Trace(){
  T tr = T(0);
  for(size_t i=0; i<nrows; i++) tr += GetVal(i, i);
  return tr;
}

template <typename T> T BDMatrix<T>::TraceProduct(BDMatrix<T>& obj){
  T tr = T(0);
  if(!(*this & obj)) return tr;
  PMatrix<T>* bl = new PMatrix<T>();
  PMatrix<T>* ob = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    obj.GetBlock(i, *ob);
    if(bl && ob) tr += bl->TraceProduct(*ob);
  }
  if(bl) delete bl;
  if(ob) delete ob;
  return tr;
}

template <typename T> T BDMatrix<T>::TraceProductSymm(BDMatrix<T>& obj){
  if(!(*this & obj)) return T(0);
  return inner_product(data.begin(), data.end(), obj.data.begin(), 0.0);
}

template <typename T> void BDMatrix<T>::Transpose(){
  PMatrix<T>* bl = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    if(bl){
      bl->Transpose();
      CopyBlock(i, bl);
    }
  }
  if(bl) delete bl;
}

template <typename T> void BDMatrix<T>::Random(){
  PMatrix<T> r = PMatrix<T>();
  for(size_t i=0; i<data.size(); i++) data[i] = r.GetRandom();
}

template <typename T>
void BDMatrix<T>::RightMult(PMatrix<T>& obj, PMatrix<T>* out) const{
  if(obj.GetNRows() != nrows) return;
  if(out){
    if(out->GetNCols() != obj.GetNCols() || out->GetNRows() != nrows) return;
    out->Zero();
  }
  else out = new PMatrix<T>(mname+"_"+obj.GetName()+"_mult",
			    obj.GetNRows(), nrows, false);
  PMatrix<T>* bl = new PMatrix<T>();
  PMatrix<T>* ob = new PMatrix<T>();
  PMatrix<T>* tb = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *bl);
    if(!bl) continue;
    size_t n = obj.GetNCols() / bl->GetNCols();
    if(obj.GetNCols() % bl->GetNCols()) n ++;
    for(size_t j=0; j<n; j++){
      size_t nc = (bl->GetNCols()*(j+1) <= obj.GetNCols()) ?
	bl->GetNCols() : (obj.GetNCols() % bl->GetNCols());
      if(ob->GetNRows() != bl->GetNRows() || ob->GetNCols() != nc){
	delete ob; delete tb;
	ob = new PMatrix<T>("tmp", bl->GetNRows(), nc, false);
	tb = new PMatrix<T>("Tmp", bl->GetNRows(), nc, false);
      }
      obj.GetBlock(cdim[i], bl->GetNCols()*j, ob);
      bl->RightMult(*ob, tb);
      out->SetBlock(cdim[i], bl->GetNCols()*j, *tb);
    }
  }
  if(bl) delete bl;
  if(ob) delete ob;
  if(tb) delete tb;
}


template <typename T>void BDMatrix<T>::Print(size_t max_bl, size_t max_size,
					     size_t prec, string lead) const{
  PMatrix<T>* block = new PMatrix<T>();
  if(max_bl == 0) max_bl = (size_t) -1;
  for(size_t i=0; i<std::min(max_bl, bdim.size()); i++){
    GetBlock(i, *block);
    if(block){
      cout << mname << " bl " << i << "  "
	   << cdim[i] << ":" << cdim[i+1] << "  ";
      block->Print(max_size, prec, lead);
    }
  }
  cout << endl;
  if(block) delete block;
}

template <typename T> void BDMatrix<T>::Print(size_t max_bl) const{
  Print(max_bl, 6, 3);
}

template <typename T>
void BDMatrix<T>::WriteHDF51d(H5::H5File &outfile, string name,
			      size_t chunk, int compression) const{
  if(name == "") name = mname;
  WriteHDF5vector<size_t>(&outfile, pblock, name + "_dim", chunk, compression);
  WriteHDF5vector<T>(&outfile, data, name   + "_data", chunk, compression);
}

template <typename T>
void BDMatrix<T>::WriteHDF5(H5::H5File &outfile, string name) const{
  if(name == "") name = mname;
  PMatrix<T>* block = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *block);
    if(block) block->WriteHDF5(outfile, name+"_"+to_string(i));
  }
  if(block) delete block;
}

template <typename T>
void BDMatrix<T>::WriteROOT(TFile* outfile, string name) const{
  if(name == "") name = mname;
  PMatrix<T>* block = new PMatrix<T>();
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *block);
    if(block) block->WriteROOT(outfile, name+"_"+to_string(i));
  }
  if(block) delete block;
}

template <typename T>
void BDMatrix<T>::WriteRdata(string outfile, string name,
			     ROOT::R::TRInterface& R){
  if(name == "") name = mname;
  PMatrix<T>* block = new PMatrix<T>();
  vector<string> bobjs, tmp;
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *block);
    if(block){
      tmp = block->AddToREnv(name+"_"+to_string(i), R);
      bobjs.insert(bobjs.end(), tmp.begin(), tmp.end());
    }
  }
  if(block) delete block;
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
    for(size_t i=0; i<bobjs.size(); i++){
      it = find(objs.begin(), objs.end(), bobjs[i]);
      if(it != objs.end()) objs.erase(it);
    }
    objs.insert(objs.end(), bobjs.begin(), bobjs.end());
    R["objs"] << objs;
    for(auto const& o : objs) R << (o + " <- nenv$" + o).c_str();
    R << ("save(list=c(objs, \""+name+"\"), file=\""+outfile+"\")").c_str();
  }
  else R << ("save(\"" + name+ "\", file=\"" + outfile + "\")").c_str();
}

template <typename T>
vector<string> BDMatrix<T>::AddToREnv(string name, ROOT::R::TRInterface& R){
  if(name == "") name = mname;
  PMatrix<T>* block = new PMatrix<T>();
  vector<string> ret;
  for(size_t i=0; i<bdim.size(); i++){
    GetBlock(i, *block);
    if(block) block->AddToREnv(name+"_"+to_string(i), R);
    ret.push_back(name+"_"+to_string(i));
  }
  if(block) delete block;
  return ret;
}  

#endif

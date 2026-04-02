#include "SparseDosageReader.hh"
#include "HDF5Util.hh"
#include <assert.h>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;
using namespace H5;

SparseDosageReader::SparseDosageReader(string infname,
				       vector<size_t> idperm,
				       double maxsize_gb){
  time_ns  = 0;
  ivariant = 0;
  auto t0 = high_resolution_clock::now();
  infile = new H5File(infname.c_str(), H5F_ACC_RDONLY);
  //get ids and permute
  ReadHDF5vector<size_t>(infile, ids, "ids");
  id_perm.resize(idperm.size());
  copy(idperm.begin(), idperm.end(), id_perm.begin());
  if(id_perm.size() > 0){
    assert(ids.size() == idperm.size());
    vector<size_t> tmp(ids.size());
    transform(id_perm.begin(), id_perm.end(), tmp.begin(),
	      [this](const size_t i){ return ids[i]; });
    copy(tmp.begin(), tmp.end(), ids.begin());
  }
  // get variant info
  ReadHDF5vector<string>(infile,   rsids,       "rsids");
  ReadHDF5vector<string>(infile,   chromosomes, "chromosomes");
  ReadHDF5vector<string>(infile,   allele0,     "allele0");
  ReadHDF5vector<string>(infile,   allele1,     "allele1");
  ReadHDF5vector<uint32_t>(infile, positions,   "positions");
  // setup hdf5 objects for dosage data
  dset = infile->openDataSet("data");
  fspace = dset.getSpace();
  fspace.getSimpleExtentDims(fdim, NULL);
  mdim[0] = (size_t) ((maxsize_gb * 1.0e9) / sizeof(uint32_t));
  mspace = DataSpace(1, mdim, mdim);
  start[0] = 0;
  offset[0] = 0;
  data.resize(mdim[0]);
  data_it = data.end();
  // initialize unpacking parameters
  delta  = 0;
  uval   = 0;
  val    = 0.0;
  index  = 0;
  lindex = 0;
  vindex = 0;
  sindex = 0;
  dbits  = 10;
  dmax   = 1 << dbits;
  dmask  = 0x3FF;
  vbits  = 20;
  vmax   = 1 << vbits;
  vmask  = 0xFFFFF << (dbits+2);
  auto t1 = high_resolution_clock::now();
  time_ns += duration_cast<nanoseconds>(t1-t0).count();
}


SparseDosageReader::~SparseDosageReader(){
  auto t0 = high_resolution_clock::now();
  fspace.close();
  mspace.close();
  dset.close();
  delete infile;
  auto t1 = high_resolution_clock::now();
  time_ns += duration_cast<nanoseconds>(t1-t0).count();
  PrintTime();
}


bool SparseDosageReader::NextChunk(){
  auto t0 = high_resolution_clock::now();
  if(offset[0] >= fdim[0]) return false;
  if(offset[0] + data.size() > fdim[0]){
    mdim[0] = fdim[0] - offset[0];
    mspace.close();
    mspace = DataSpace(1, mdim, mdim);
    data.resize(mdim[0]);
  }
  fspace.selectHyperslab(H5S_SELECT_SET, mdim, offset);
  mspace.selectHyperslab(H5S_SELECT_SET, mdim, start);
  dset.read(&data.front(), PredType::NATIVE_UINT32, mspace, fspace);
  data_it = data.begin();
  auto t1 = high_resolution_clock::now();
  time_ns += duration_cast<nanoseconds>(t1-t0).count();
  return true;
}


bool SparseDosageReader::NextVariant(PMatrix<double>* G){
  assert(G->GetNData() == ids.size() &&
	 (G->GetNRows() == 1 || G->GetNCols() == 1));
  G->Zero();
  if(ivariant >= rsids.size()) return false;
  if(data_it == data.end()) if(!NextChunk()) return false;
  auto t0 = high_resolution_clock::now();
  bool valid_chunk = true;
  while(valid_chunk){
    while(data_it != data.end()){
      dval = *data_it;
      delta = dval & dmask;
      if(dval      & (dmax << 1)) val = 2.0;
      else if(dval &  dmax)       val = 1.0;
      else                        val = 0.0;
      uval = (val == 2.0) ? 0 : ((dval & vmask) >> (dbits+2));
      val += uval / (double) vmax;
      index += delta;
      vindex = index / ids.size();
      sindex = index % ids.size();
      if(lindex != vindex){
	lindex = vindex;
	index -= delta;
	ivariant ++;
	if(id_perm.size() > 0) G->Permute(id_perm);
	auto t1 = high_resolution_clock::now();
	time_ns += duration_cast<nanoseconds>(t1-t0).count();
	return true;
      }
      G->SetVal(sindex, val);
      data_it ++;
      if(data_it == data.end()) offset[0] += data.size();
    }
    valid_chunk = NextChunk();
  }
  ivariant ++;
  if(id_perm.size() > 0) G->Permute(id_perm);
  auto t1 = high_resolution_clock::now();
  time_ns += duration_cast<nanoseconds>(t1-t0).count();
  return true;
}


void SparseDosageReader::PrintTime(){
  cout << "SparseDosageReader: read " << ivariant << " of " << rsids.size()
       << " variants in " << GetTotalTimeSec() << " s, "
       << GetnsPerVariant() << " ns/variant" << endl;
}


void SparseDosageReader::WriteHeader(H5File* outfile, string base){
  if(!outfile) return;
  WriteHDF5vector<size_t>(outfile,   ids,         base+"ids");
  WriteHDF5vector<string>(outfile,   rsids,       base+"rsids");
  WriteHDF5vector<string>(outfile,   chromosomes, base+"chromosomes");
  WriteHDF5vector<string>(outfile,   allele0,     base+"allele0");
  WriteHDF5vector<string>(outfile,   allele1,     base+"allele1");
  WriteHDF5vector<uint32_t>(outfile, positions,   base+"positions");
}

#include "PMatrix.hh"
#include "ReadBGEN.hh"
#include "HDF5Util.hh"
#include <getopt.h>
#include <assert.h>
#include <chrono>
#include <H5Cpp.h>

using namespace std;
using namespace std::chrono;
using namespace H5;

int main(int argc, char* argv[]){

  // option handling
  string infilename  = "";
  string hdf5outfile = "";
  string hdf5infile  = "";
  int clevel         = 1;
  double g_maxsize  = (1 << 30) / 1.0e9;
  static struct option opts[]{
    {"infile",   required_argument, NULL, 'i'},
    {"gbchunk",  required_argument, NULL, 'G'},
    {"hdf5in",   required_argument, NULL, 'h'},
    {"compress", required_argument, NULL, 'c'}};
  int opt = getopt_long(argc, argv, "i:G:h:c:", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'i': infilename  = string(optarg); break;
    case 'G': g_maxsize   =   stod(optarg); break;
    case 'h': hdf5infile  = string(optarg); break;
    case 'c': clevel      = std::max(0, std::min(9, atoi(optarg))); break;
    default:
      cout << "unrecognized option -" << opt << endl;
      return 1;
    }
    opt = getopt_long(argc, argv, "i:G:h:c:", opts, NULL);
  }

  // values used in both writing and reading
  const size_t max_size = (size_t) ((g_maxsize * 1.0e9) / sizeof(uint32_t));
  hsize_t maxdim[] = {H5S_UNLIMITED}, start[] = {0};
  StrType strType(PredType::C_S1, H5T_VARIABLE);
  vector<size_t> ids;
  vector<uint32_t> positions;
  vector<string> rsids, chromosomes, allele0, allele1;
  size_t offset = 0, lindex = 0;
  vector<uint32_t> data, overflow;
  if(hdf5infile == ""){
    auto t0 = high_resolution_clock::now();
    assert(infilename != "");
    hdf5outfile = infilename.substr(0, infilename.find('.')) + ".hdf5";
    // get the bgen parser and dimensions
    BgenParser parser(infilename);
    parser.summarise(cerr);
    const size_t nind = parser.number_of_samples();
    const size_t nvar = parser.number_of_variants();
    ids.resize(nind);
    size_t nid = 0;
    // get the individual ids and write them to the hdf5 output file
    parser.get_sample_ids([&ids,&nid](string const& id){
      ids[nid] = (size_t) stoi(id.substr(0, id.find("_"))); nid++; });
    H5File* hdf5_file = new H5File(hdf5outfile.c_str(), H5F_ACC_TRUNC);
    WriteHDF5vector<size_t>(hdf5_file, ids, "ids");
    NextSparseBGENDosages(parser, offset, lindex,
			  chromosomes, positions, allele0, allele1,
			  rsids, data,max_size);
    if(data.size() > max_size){
      overflow.resize(data.size() - max_size);
      copy(data.begin()+max_size, data.end(), overflow.begin());
      data.resize(max_size);
    }
    // set up the dataspace for the dosage values and write to file
    hsize_t dim[] = {data.size()};
    hsize_t chunk[] = {max_size};
    DataSpace fspace(1, dim, maxdim);
    DSetCreatPropList plist;
    uint32_t fval = 0;
    plist.setFillValue(PredType::NATIVE_UINT32, &fval);
    plist.setChunk(1, chunk);
    plist.setDeflate(clevel);
    DataSet dset = hdf5_file->createDataSet("data", PredType::NATIVE_UINT32,
					    fspace, plist);
    dset.write(&data.front(), PredType::NATIVE_UINT32);
    // move the overflow values to the data
    size_t foffset = dim[0];
    data.resize(overflow.size());
    copy(overflow.begin(), overflow.end(), data.begin());
    // get the remaining blocks of data and write to file
    while(NextSparseBGENDosages(parser, offset, lindex,
				chromosomes, positions, allele0, allele1,
				rsids, data, max_size)){
      if(data.size() > max_size){
	overflow.resize(data.size() - max_size);
	copy(data.begin()+max_size, data.end(), overflow.begin());
	data.resize(max_size);
      }
      else overflow.resize(0);
      hsize_t count[] = {data.size()};
      hsize_t off[]   = {foffset};
      hsize_t fdim[]  = {foffset + data.size()};
      dset.extend(fdim);
      fspace.close();
      fspace = DataSpace(1, fdim, maxdim);
      fspace.selectHyperslab(H5S_SELECT_SET, count, off);
      DataSpace mspace(1, count, maxdim);
      mspace.selectHyperslab(H5S_SELECT_SET, count, start);
      dset.write(&data.front(), PredType::NATIVE_UINT32, mspace, fspace);
      mspace.close();
      foffset += data.size();
      data.resize(overflow.size());
      copy(overflow.begin(), overflow.end(), data.begin());
    }
    // write the last overflow values
    if(data.size() > 0){
      hsize_t count[] = {data.size()};
      hsize_t off[]   = {foffset};
      hsize_t fdim[]  = {foffset + data.size()};
      dset.extend(fdim);
      fspace.close();
      fspace = DataSpace(1, fdim, maxdim);
      fspace.selectHyperslab(H5S_SELECT_SET, count, off);
      DataSpace mspace(1, count, maxdim);
      mspace.selectHyperslab(H5S_SELECT_SET, count, start);
      dset.write(&data.front(), PredType::NATIVE_UINT32, mspace, fspace);
      mspace.close();
      foffset += data.size();
    }
    fspace.close();
    dset.close();
    // write the rsids
    WriteHDF5vector<string>(hdf5_file, rsids, "rsids");
    WriteHDF5vector<string>(hdf5_file, chromosomes, "chromosomes");
    WriteHDF5vector<uint32_t>(hdf5_file, positions, "positions");
    WriteHDF5vector<string>(hdf5_file, allele0, "allele0");
    WriteHDF5vector<string>(hdf5_file, allele1, "allele1");
    delete hdf5_file;
    auto t1 = high_resolution_clock::now();
    double us = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    cout << "read " << nvar << " variants in " << us*1.0e-6 << " s"
	 << ", " << us/nvar << " us/variant" << endl;
  }
  else{
    auto t0 = high_resolution_clock::now();
    H5File* hdf5_file = new H5File(hdf5infile.c_str(), H5F_ACC_RDONLY);
    // read the individual ids
    ReadHDF5vector<size_t>(hdf5_file, ids, "ids");
    const size_t nids = ids.size();
    cout << "ids:" << endl;
    for(size_t i=0; i<10; i++) cout << "  " << ids[i] << endl;
    cout << "  ..." << endl;
    // read the variant ids
    ReadHDF5vector<string>(hdf5_file, rsids, "rsids");
    cout <<"rsids:" << endl;
    cout << rsids.size() << endl;
    for(size_t i=0; i<10; i++) cout << "  " << rsids[i] << endl;
    cout << "  ..." << endl;
    // setup the data space for the dosage data
    DataSet dset = hdf5_file->openDataSet("data");
    DataSpace fspace = dset.getSpace();
    hsize_t fdim[] = {1};
    fspace.getSimpleExtentDims(fdim, NULL);
    hsize_t dim[] = {max_size};
    DataSpace mspace(1, dim, maxdim);
    hsize_t off[] = {offset};
    // set data size and create PMatrix to temporarily store each variant
    data.resize(max_size);
    size_t ndata = data.size();
    PMatrix<double>* G = new PMatrix<double>("G", ids.size(), 1, false);
    // parameters for unpacking the data
    uint16_t delta = 0;
    uint32_t uval = 0;
    double val    = 0.0;
    size_t index  = 0, lindex = 0, vindex = 0, sindex = 0;
    const uint32_t dbits = 10, dmax = (1 << dbits), dmask = 0x3FF;
    const uint32_t vbits = 20, vmax = (1 << vbits), vmask = 0xFFFFF << (dbits + 2);
    // read each chunk and unpack each value
    while(offset < fdim[0]){
      if(offset + max_size > fdim[0]){
	dim[0] = fdim[0] - offset;
	mspace.close();
	mspace = DataSpace(1, dim, maxdim);
	data.resize(dim[0]);
	ndata = data.size();
      }
      fspace.selectHyperslab(H5S_SELECT_SET, dim, off);
      mspace.selectHyperslab(H5S_SELECT_SET, dim, start);
      dset.read(&data.front(), PredType::NATIVE_UINT32, mspace, fspace);
      for(auto const& d : data){
	delta = d & dmask;
	if(d      & (dmax << 2)) val = 2.0;
	else if(d & (dmax << 1)) val = 1.0;
	else                     val = 0.0;
	uval = (val == 2.0) ? 0 : ((d & vmask) >> (dbits+2));
	val += uval / (double) vmax;
	index += delta;
	vindex = index / nids;
	sindex = index % nids;
	if(lindex != vindex){
	  if(lindex % 100000 == 0) G->Print(20, 8);
	  G->Zero();
	  lindex = vindex;
	}
	G->SetVal(sindex, val);
      }
      offset += ndata;
      off[0] = offset;
    }
    // cleanup and print execution time
    fspace.close();
    mspace.close();
    dset.close();
    delete hdf5_file;
    auto t1 = high_resolution_clock::now();
    double us = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    cout << "read " << vindex+1 << " variants in " << us*1.0e-6 << " s"
	 << ", " << us/(vindex+1) << " us/variant" << endl;
  }
  
  return 0;
}

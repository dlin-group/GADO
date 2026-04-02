#include "PMatrix.hh"
#include "BDMatrix.hh"
#include "GHAModel.hh"
#include "GHAIO.hh"
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#include <cblas.h>
#ifdef __CUDA__
#include <cublas_v2.h>
#endif

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

  size_t bsize       = 3;
  string ghpath      = "../grmfit/gh/data";
  string ghbase      = "hermite";
  size_t ghdim       = 2;
  size_t ghorder     = 12;
  bool   ghstep      = false;
  size_t threads     = 4;
  bool   b_last      = true;
  bool   b_fullnr    = true;
  bool   verbose     = false;
  static struct option opts[]{
    {"help",             no_argument, NULL, 'h'},
    {"bsize",      required_argument, NULL, 'b'},
    {"ghpath",     required_argument, NULL, 'g'},
    {"ghbase",     required_argument, NULL, 'G'},
    {"ndim",       required_argument, NULL, 'n'},
    {"ndimsi",     required_argument, NULL, 'N'},
    {"order",      required_argument, NULL, 'o'},
    {"orderstep",        no_argument, NULL, 'O'},
    {"threads",    required_argument, NULL, 't'},
    {"zerob",            no_argument, NULL, 'z'},
    {"sstepb",           no_argument, NULL, 's'},
    {"verbose",          no_argument, NULL, 'v'}};
  int opt = getopt_long(argc, argv, "hb:g:G:n:o:Ot:zsv", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                            << endl
	   << "  -b block size"                      << endl
	   << "  -g path to GH data"                 << endl
	   << "  -G base of GH files"                << endl
	   << "  -n dimension of GH approx"          << endl
	   << "  -o order     of GH approx"          << endl
	   << "  -O step GH order"                   << endl
	   << "  -t number of threads"               << endl
	   << "  -z b=0 at each step"                << endl
	   << "  -s single step NR for b"            << endl
	   << "  -v verbose mode"                    << endl
	   << "  -r write to Rdata file"             << endl;
      return 0;
    case 'b': bsize       =   atoi(optarg); break;
    case 'g': ghpath      = string(optarg); break;
    case 'G': ghbase      = string(optarg); break;
    case 'n': ghdim       =   atoi(optarg); break;
    case 'o': ghorder     =   atoi(optarg); break;
    case 'O': ghstep      =           true; break;
    case 't': threads     =   atoi(optarg); break;
    case 'z': b_last      =          false; break;
    case 's': b_fullnr    =          false; break;
    case 'v': verbose     =           true; break;
    default:
      cout << "unrecoginized option -" << opt << endl;
      return 1;
    }
    opt = getopt_long(argc, argv, "hb:g:G:n:o:Ot:zsv", opts, NULL);
  }


  vector<double> vblock;
  if(bsize == 3)
    vblock = {1.0, 0.0, 0.5,
              0.0, 1.0, 0.5,
              0.5, 0.5, 1.0};
  else if(bsize == 10)
    vblock = {
      1.00, 0.00, 0.50, 0.0, 0.50, 0.0, 0.250, 0.250, 0.250, 0.250, 
      0.00, 1.00, 0.50, 0.0, 0.50, 0.0, 0.250, 0.250, 0.250, 0.250, 
      0.50, 0.50, 1.00, 0.0, 0.50, 0.0, 0.500, 0.500, 0.250, 0.250, 
      0.00, 0.00, 0.00, 1.0, 0.00, 0.0, 0.500, 0.500, 0.000, 0.000, 
      0.50, 0.50, 0.50, 0.0, 1.00, 0.0, 0.250, 0.250, 0.500, 0.500, 
      0.00, 0.00, 0.00, 0.0, 0.00, 1.0, 0.000, 0.000, 0.500, 0.500, 
      0.25, 0.25, 0.50, 0.5, 0.25, 0.0, 1.000, 0.500, 0.125, 0.125, 
      0.25, 0.25, 0.50, 0.5, 0.25, 0.0, 0.500, 1.000, 0.125, 0.125, 
      0.25, 0.25, 0.25, 0.0, 0.50, 0.5, 0.125, 0.125, 1.000, 0.500, 
      0.25, 0.25, 0.25, 0.0, 0.50, 0.5, 0.125, 0.125, 0.500, 1.000};
  else{
    cout << "block size must be 3 or 10" << endl;
    return 1;
  }

  string f0 = "../grmfit/data/compare_";
  f0 += to_string(bsize) + "_" + to_string(bsize) + ".csv";
  PMatrix<double>*     T = new PMatrix<double>("T", f0,
					      vector<size_t>(1, 1));
  if(T->GetNData() % bsize != 0){
    cout << "Number of data entries " << T->GetNData()
	 << " is not divisible by the block size " << bsize << endl;
    delete T;
    return 2;
  }
  PMatrix<double>* delta = new PMatrix<double>("delta", f0,
					       vector<size_t>(1, 2));
  PMatrix<double>*     X = new PMatrix<double>("X", f0,
					       vector<size_t>({3, 4}));
  PMatrix<double>*     G0 = new PMatrix<double>("G0", f0,
						vector<size_t>(1, 6));
  PMatrix<double>*     G1 = new PMatrix<double>("G1", f0,
  						vector<size_t>(1, 7));

  PMatrix<double>* block = new PMatrix<double>("block", bsize, bsize, false);
  block->SetData(vblock);
  const size_t nblocks = T->GetNData() / bsize;
  BDMatrix<double>* psi = new BDMatrix<double>("psi");
  for(size_t i=0; i<nblocks; i++) psi->AppendBlock(block);
  delete block;
  
  cout << "sorting input data by block size, permuting input data..." << endl;
  auto t0 = high_resolution_clock::now();
  psi->SortBlocks();
  vector<size_t> permutation = psi->GetIndexPermutation();
  T->PermuteRows(permutation);
  delta->PermuteRows(permutation);
  X->PermuteRows(permutation);
  G0->PermuteRows(permutation);
  auto t1 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;

  GHAModel* model = new GHAModel(ghpath, ghbase, ghdim, ghorder, ghstep,
				 T, delta, X, psi);
  model->SetEpsilon(1.0e-5);
  model->SetBLast(b_last);
  model->SetBFullNR(b_fullnr);
  model->SetVerbose(verbose);
  openblas_set_num_threads(threads);
  model->Solve();
  //model->WriteNullModelHDF5("nm.hdf5");
  //model->ReadNullModelHDF5("nm.hdf5");
  model->ComputeScoreInfo();
  //model->WriteScoreInfoHDF5("si.hdf5");
  //model->ReadScoreInfoHDF5("si.hdf5");
  #ifdef __CUDA__
  cublasHandle_t chandle;
  cublasCreate_v2(&chandle);
  model->SetCublasHandle(chandle);
  #endif
  model->SetG(G0);
  model->ComputeScoreTest(true);
  model->SetG(G1);
  model->ComputeScoreTest(true);
  
  delete T;
  delete delta;
  delete X;
  delete psi;
  delete model;

  return 0;
}

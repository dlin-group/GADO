#include "PMatrix.hh"
#include "GMatrix.hh"
#include "BGMatrix.hh"
#include "GHAModelGPU.hh"
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#include <cblas-openblas.h>
#include <cublas_v2.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

  string f0       = "/home/tcald/bios/lin/grm/dat_10_10.csv";
  string f1       = "";//"/home/tcald/bios/lin/grm/psi_10_10.csv";
  string ghpath   = "/home/tcald/bios/hermite/data";
  string ghbase   = "hermite";
  size_t ghdim    = 2;
  size_t ghorder  = 16;
  size_t threads  = 4;
  bool   b_last   = true;
  bool   b_fullnr = true;
  bool   verbose  = false;
  static struct option opts[]{
    {"help",          no_argument, NULL, 'h'},
    {"data",    required_argument, NULL, 'd'},
    {"psi",     required_argument, NULL, 'p'},
    {"ghpath",  required_argument, NULL, 'g'},
    {"ghbase",  required_argument, NULL, 'G'},
    {"ndim",    required_argument, NULL, 'n'},
    {"order",   required_argument, NULL, 'o'},
    {"threads", required_argument, NULL, 't'},
    {"zerob",         no_argument, NULL, 'z'},
    {"sstepb",        no_argument, NULL, 's'},
    {"verbose",       no_argument, NULL, 'v'}};
  int opt = getopt_long(argc, argv, "hd:p:g:G:n:o:t:zsv", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                   << endl
	   << "  -d data csv file"          << endl
	   << "  -p psi  csv file"          << endl
	   << "  -g path to GH data"        << endl
	   << "  -G base of GH files"       << endl
	   << "  -n dimension of GH approx" << endl
	   << "  -o order     of GH approx" << endl
	   << "  -t number of threads"      << endl
	   << "  -z b=0 at each step"       << endl
	   << "  -s single step NR for b"   << endl
	   << "  -v verbose mode"           << endl;
      return 0;
    case 'd': f0       = string(optarg); break;
    case 'p': f1       = string(optarg); break;
    case 'g': ghpath   = string(optarg); break;
    case 'G': ghbase   = string(optarg); break;
    case 'n': ghdim    =   atoi(optarg); break;
    case 'o': ghorder  =   atoi(optarg); break;
    case 't': threads  =   atoi(optarg); break;
    case 'z': b_last   =          false; break;
    case 's': b_fullnr =          false; break;
    case 'v': verbose  =           true; break;
    default:
      cout << "unrecoginized option -" << opt << endl;
      return 1;
    }
    opt = getopt_long(argc, argv, "hd:p:g:G:n:o:t:zsv", opts, NULL);
  }

  vector<double> vblock({
      1.0e+0,0.0e+0,5.0e-1,0.0e+0,5.0e-1,0.0e+0,2.5e-1,2.5e-1,2.5e-1,2.5e-1,
      0.0e+0,1.0e+0,5.0e-1,0.0e+0,5.0e-1,0.0e+0,2.5e-1,2.5e-1,2.5e-1,2.5e-1,
      5.0e-1,5.0e-1,1.0e+0,0.0e+0,5.0e-1,0.0e+0,5.0e-1,5.0e-1,2.5e-1,2.5e-1,
      0.0e+0,0.0e+0,0.0e+0,1.0e+0,0.0e+0,0.0e+0,5.0e-1,5.0e-1,0.0e+0,0.0e+0,
      5.0e-1,5.0e-1,5.0e-1,0.0e+0,1.0e+0,0.0e+0,2.5e-1,2.5e-1,5.0e-1,5.0e-1,
      0.0e+0,0.0e+0,0.0e+0,0.0e+0,0.0e+0,1.0e+0,0.0e+0,0.0e+0,5.0e-1,5.0e-1,
      2.5e-1,2.5e-1,5.0e-1,5.0e-1,2.5e-1,0.0e+0,1.0e+0,5.0e-1,1.25e-1,1.25e-1,
      2.5e-1,2.5e-1,5.0e-1,5.0e-1,2.5e-1,0.0e+0,5.0e-1,1.0e+0,1.25e-1,1.25e-1,
      2.5e-1,2.5e-1,2.5e-1,0.0e+0,5.0e-1,5.0e-1,1.25e-1,1.25e-1,1.0e+0,5.0e-1,
      2.5e-1,2.5e-1,2.5e-1,0.0e+0,5.0e-1,5.0e-1,1.25e-1,1.25e-1,5.0e-1,1.0e+0});

  PMatrix<double>*     T_tmp = new PMatrix<double>("T_tmp", f0,
						   vector<size_t>(1, 1));
  PMatrix<double>* delta_tmp = new PMatrix<double>("delta_tmp", f0,
						   vector<size_t>(1, 2));
  PMatrix<double>*     X_tmp = new PMatrix<double>("X_tmp", f0,
						   vector<size_t>({3, 4}));
  
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  GMatrix<double>*     T = new GMatrix<double>("T",     *T_tmp,     handle);
  GMatrix<double>* delta = new GMatrix<double>("delta", *delta_tmp, handle);
  GMatrix<double>*     X = new GMatrix<double>("X",     *X_tmp,     handle);

  BGMatrix<double>* psi;
  if(f1 == ""){
    PMatrix<double>* block = new PMatrix<double>("block", 10, 10, false);
    block->SetData(vblock);
    psi = new BGMatrix<double>("psi", handle);
    for(size_t i=0; i<500; i++) psi->AppendBlock(block);
    delete block;
  }
  else{
    PMatrix<double>* Psi = new PMatrix<double>("Psi", f1,
					       vector<size_t>(0),
					       vector<size_t>(1, 0));
    BDMatrix<double>* psi_tmp = new BDMatrix<double>("psi", *Psi, 1.0e-12);
    psi = new BGMatrix<double>("psi", *psi_tmp, handle);
    delete Psi;
    delete psi_tmp;
  }
  psi->CloseToAppend();
  
  vector<size_t> permutation = psi->GetIndexPermutation();
  T->PermuteRows(permutation);
  delta->PermuteRows(permutation);
  X->PermuteRows(permutation);
  T->CopyToDevice();
  delta->CopyToDevice();
  X->CopyToDevice();

  GHAModelGPU* model = new GHAModelGPU(ghpath, ghbase, ghdim, ghorder,
				       T, delta, X, psi, handle);
  model->SetEpsilon(1.0e-5);
  model->SetBLast(b_last);
  model->SetBFullNR(b_fullnr);
  model->SetVerbose(verbose);
  openblas_set_num_threads(threads);
  model->Solve();

  cublasDestroy_v2(handle);
  delete T;
  delete delta;
  delete X;
  delete psi;
  delete model;

  return 0;
}

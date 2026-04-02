#include "PMatrix.hh"
#include "BDMatrix.hh"
#include "GHAModel.hh"
#include <Math/GSLRndmEngines.h>
#include <TRandom.h>
#include <H5Cpp.h>
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int argc,  char* argv[]){

  size_t nthreads = 12;
  string ukbpath  = "";
  string ghpath   = "";
  size_t ghorder  = 4;
  bool ghstep     = false;
  int seed        = -1;
  bool verbose    = false;
  bool b_full     = false;
  bool b_zero     = false;
  double tausim   = 1.0;
  double lambda   = 0.3;
  double epsilon  = 1.0e-5;
  vector<double> betasim;
  static struct option opts[]{
    {"help",           no_argument, NULL, 'h'},
    {"nthreads", required_argument, NULL, 'j'},
    {"ukbpath",  required_argument, NULL, 'u'},
    {"pathgh",   required_argument, NULL, 'p'},
    {"ghorder",  required_argument, NULL, 'g'},
    {"ghstep",         no_argument, NULL, 'G'},
    {"seed",     required_argument, NULL, 's'},
    {"verbose",        no_argument, NULL, 'v'},
    {"fullb",          no_argument, NULL, 'f'},
    {"zerob",          no_argument, NULL, 'z'},
    {"tau",      required_argument, NULL, 't'},
    {"lambda",   required_argument, NULL, 'l'},
    {"beta",     required_argument, NULL, 'b'},
    {"epsilon",  required_argument, NULL, 'e'}};
  int opt = getopt_long(argc, argv, "hj:u:p:g:Gs:vfzt:l:b:e:", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                   << endl
	   << "  -u path to ukb grm matrix" << endl
	   << "  -j number of threads"      << endl
	   << "  -p path to GH files"       << endl
	   << "  -g order of GH approx"     << endl
	   << "  -G step GH order"          << endl
	   << "  -s random seed"            << endl
	   << "  -v verbose printing"       << endl
	   << "  -f full NR for b"          << endl
	   << "  -z zero b at each iter"    << endl
	   << "  -t simulated tau"          << endl
	   << "  -l simulated lambda"       << endl
	   << "  -b append value for beta"  << endl
	   << "  -e epsilon convergence"    << endl;
      break;
    case 'u': ukbpath  = string(optarg); break;
    case 'j': nthreads =   atoi(optarg); break;
    case 'p': ghpath   = string(optarg); break;
    case 'g': ghorder  =   atoi(optarg); break;
    case 'G': ghstep   =           true; break;
    case 's': seed     =   atoi(optarg); break;
    case 'v': verbose  =           true; break;
    case 'f': b_full   =           true; break;
    case 'z': b_zero   =           true; break;
    case 't': tausim   =   atof(optarg); break;
    case 'l': lambda   =   atof(optarg); break;
    case 'e': epsilon  =   stod(optarg); break;
    case 'b': betasim.push_back(atof(optarg)); break;
    default:
            cout << "skipping unrecognized option " << "-" << opt << endl;
      break;
    }
    opt = getopt_long(argc, argv, "hj:u:p:g:Gs:vfzt:l:b:e:", opts, NULL);
  }
  if(ukbpath == ""){
    cout << "must specify path to UKB GRM matrix with -u" << endl;
    return 1;
  }
  if(ghpath == ""){
    cout << "must specify path to GH files with -p" << endl;
  }
  assert(tausim > 0.0);
  assert(epsilon > 0.0 && epsilon < 1.0);
  assert(betasim.size() == 2);
  assert(ghorder < 129);
  const size_t npar = betasim.size();
  openblas_set_num_threads(nthreads);

  //bool user_seed = true;
  if(seed < 0){
    gRandom->SetSeed();
    seed = gRandom->GetSeed();
    //user_seed = false;
  }
  else gRandom->SetSeed(seed);
  ROOT::Math::GSLRandomEngine gsl_rand;
  gsl_rand.Initialize();
  gsl_rand.SetSeed(seed);

  cout << "reading input grm..." << endl;
  auto t0 = high_resolution_clock::now();
  H5::H5File ukbfile(ukbpath.c_str(), H5F_ACC_RDONLY);
  BDMatrix<double>* psi = new BDMatrix<double>("psi", "block_", ukbfile);
  auto t1 = high_resolution_clock::now();
  cout << "  read " << psi->GetNBlocks() << " blocks in "
       << scientific << setprecision(3)
       << duration_cast<nanoseconds>(t1-t0).count()/1.0e9 << " s" << endl;

  PMatrix<double>*       T = new PMatrix<double>("T",
						 psi->GetNRows(), 1, false);
  PMatrix<double>*   delta = new PMatrix<double>("delta",
						 psi->GetNRows(), 1, false);
  PMatrix<double>*       X = new PMatrix<double>("X",
						psi->GetNRows(), npar, false);

  cout << "sorting input data by block size, permuting input data..." << endl;
  t0 = high_resolution_clock::now();
  psi->SortBlocks();
  vector<size_t> permutation = psi->GetIndexPermutation();
  T->PermuteRows(permutation);
  delta->PermuteRows(permutation);
  X->PermuteRows(permutation);
  t1 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;

  BDMatrix<double>* psitmp = new BDMatrix<double>("psitmp", *psi);
  (*psitmp) *= tausim;

  cout << "generating simulated dataset..." << endl;
  t0 = high_resolution_clock::now();
  size_t offset = 0;
  for(size_t i=0; i<psitmp->GetNBlocks(); i++){
    const size_t nrows = psitmp->GetBlockRows(i);
    vector<double> m(nrows, 0.0), b(nrows, 0.0);
    gsl_rand.GaussianND(nrows, &m.front(),
			&psitmp->GetData().front()+psitmp->GetBlockDataOffset(i),
			&b.front());
    size_t k = offset;
    for(size_t j=0; j<nrows; (j++, k++)){
      X->SetVal(k, 0, gsl_rand()<=0.5 ? 0.0 : 1.0);
      X->SetVal(k, 1, gsl_rand.Gaussian(1.0));
      double c = std::min(3.0, 4*gsl_rand());
      double t = -log(gsl_rand()) / lambda;
      t /= exp(X->GetVal(k, 0)*betasim[0] + X->GetVal(k, 1)*betasim[1] + b[j]);
      T->SetVal(k, std::min(c, t));
      delta->SetVal(k, c >= t ? 1.0 : 0.0);
    }
    offset += nrows;
  }
  t1 = high_resolution_clock::now();
  cout << "  " <<
    duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;
  delete psitmp;

  GHAModel* model = new GHAModel(ghpath, "hermite", 2, ghorder, ghstep,
				 T, delta, X, psi);
  model->SetVerbose(verbose);
  model->SetBFullNR(b_full);
  model->SetBLast(!b_zero);
  model->Solve();
  
  
  delete psi;
  delete T;
  delete delta;
  delete X;
  delete model;
  
  return 0;
}

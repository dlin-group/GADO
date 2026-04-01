#include "PMatrix.hh"
#include "BDMatrix.hh"
#include "GHAModel.hh"
#include "GHAIO.hh"
#include <H5Cpp.h>
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <random>
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

  string ghpath            = "../grmfit/gh/data";
  string ghbase            = "hermite";
  size_t ghdim             = 2;
  size_t ghorder           = 12;
  bool   ghstep            = false;
  size_t threads           = 4;
  size_t limitvar          = SIZE_MAX;
  string fitnmin           = "";
  string siin              = "";
  string testout           = "";
  string cvfile            = "";
  size_t ncovariates       =  1;
  string grmfile           = "";
  string varfile           = "";
  size_t nvariants         = 0;
  double epsilon_vr        = 1.0e-4;
  double spa_thresh        = numeric_limits<double>::max();
  bool cache_SI            = true;
  bool bnorm_ugamma        = false;
  bool utransform_G        = false;
  bool verbose             = false;
  const bool   b_last      = true;
  const bool   b_fullnr    = false;
  const double epsilon     = 1.0e-5;
  static struct option opts[]{
    {"help",             no_argument, NULL, 'h'},
    {"ghpath",     required_argument, NULL, 'g'},
    {"ghbase",     required_argument, NULL, 'G'},
    {"ndim",       required_argument, NULL, 'n'},
    {"order",      required_argument, NULL, 'o'},
    {"orderstep",        no_argument, NULL, 'O'},
    {"jthreads",   required_argument, NULL, 'j'},
    {"limitvar",   required_argument, NULL, 'l'},
    {"fitnmin",    required_argument, NULL, 'F'},
    {"siin",       required_argument, NULL, 'S'},
    {"testout",    required_argument, NULL, 't'},
    {"cvfile",     required_argument, NULL, 'c'},
    {"ncovariates",required_argument, NULL, 'C'},
    {"grmfile",    required_argument, NULL, 'm'},
    {"varfile",    required_argument, NULL, 'r'},
    {"nvariants",  required_argument, NULL, 'B'},
    {"epsilonvr",  required_argument, NULL, 'E'},
    {"spathresh",  required_argument, NULL, 'p'},
    {"donotcache",       no_argument, NULL, 'd'},
    {"ugammabnorm",      no_argument, NULL, 'u'},
    {"utransformG",      no_argument, NULL, 'U'},
    {"verbose",          no_argument, NULL, 'v'}};
  int opt = getopt_long(argc, argv,
			"hg:G:n:o:Oj:l:F:S:t:c:C:m:b:B:E:p:duUv",
			opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                            << endl
	   << "  -g path to GH data"                 << endl
	   << "  -G base of GH files"                << endl
	   << "  -n dimension of GH approx"          << endl
	   << "  -o order     of GH approx"          << endl
	   << "  -O step GH order"                   << endl
	   << "  -j number of threads"               << endl
	   << "  -l limit number variants to read"   << endl
	   << "  -F null model fit input"            << endl
	   << "  -S score/information input"         << endl
	   << "  -t score test output"               << endl
	   << "  -c covariate txt file"              << endl
	   << "  -C number of covariates"            << endl
	   << "  -m grm matrix Rdata file"           << endl
	   << "  -r csv file to read variants from"  << endl
	   << "  -B number of variants in csv file"  << endl
	   << "  -E epsilon for variance ratio"      << endl
	   << "  -p use SPA when abs(score) < value" << endl
	   << "  -d do not cache Stheta and Igamma"  << endl
	   << "  -u normalize Ugamma by block size"  << endl
	   << "  -U apply unitary transform to G"    << endl
	   << "  -v verbose mode"                    << endl;
      return 0;
    case 'g': ghpath      = string(optarg); break;
    case 'G': ghbase      = string(optarg); break;
    case 'n': ghdim       =   atoi(optarg); break;
    case 'o': ghorder     =   atoi(optarg); break;
    case 'O': ghstep      =true; ghorder=4; break;
    case 'j': threads     =   atoi(optarg); break;
    case 'l': limitvar    =   atoi(optarg); break;
    case 'F': fitnmin     = string(optarg); break;
    case 'S': siin        = string(optarg); break;
    case 't': testout     = string(optarg); break;
    case 'c': cvfile      = string(optarg); break;
    case 'C': ncovariates =   atoi(optarg); break;
    case 'm': grmfile     = string(optarg); break;
    case 'r': varfile     = string(optarg); break;
    case 'B': nvariants   =   atoi(optarg); break;
    case 'E': epsilon_vr  =   stod(optarg); break;
    case 'p': spa_thresh  =   stod(optarg); break;
    case 'd': cache_SI    =          false; break;
    case 'u': bnorm_ugamma=           true; break;
    case 'U': utransform_G=           true; break;
    case 'v': verbose     =           true; break;
    default:
      cout << "unrecoginized option -" << opt << endl;
      return 1;
    }
    opt = getopt_long(argc, argv,
		      "hg:G:n:o:Oj:l:F:S:t:c:C:m:r:B:E:p:duUv",
		      opts, NULL);
  }

  assert(cvfile  != "");
  assert(grmfile != "");
  assert(fitnmin == "" || siin == "");
  assert(testout != "" || nvariants > 0);

  cout << "reading input data..." << endl;
  auto t0 = high_resolution_clock::now();
  PMatrix<double>*     T = new PMatrix<double>("T", cvfile,
					       vector<size_t>({2}));
  PMatrix<double>* delta = new PMatrix<double>("delta", cvfile,
					       vector<size_t>({1}));
  vector<size_t> cvindex(ncovariates);
  iota(cvindex.begin(), cvindex.end(), 3);
  PMatrix<double>*     X = new PMatrix<double>("X", cvfile, cvindex);
  H5::H5File grmh5(grmfile.c_str(), H5F_ACC_RDONLY);
  PMatrix<double>* grm = new PMatrix<double>("sGRM", grmh5);
  grmh5.close();
  BDMatrix<double>*  psi = new BDMatrix<double>("psi");
  psi->AppendBlock(grm);
  delete grm;
  auto t1 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;
  
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

  GHAModel* model = new GHAModel(ghpath, ghbase, ghdim, ghorder, ghstep,
				 T, delta, X, psi);
  model->SetEpsilon(epsilon);
  model->SetBLast(b_last);
  model->SetBFullNR(b_fullnr);
  model->SetVerbose(verbose);
  model->SetGHStepIterConv(50);
  model->SetUseVarRatio(true);
  model->SetVarRatioEpsilon(epsilon_vr);
  model->SetSPAThreshold(spa_thresh);
  model->SetCacheSthetaIgamma(cache_SI);
  model->SetBNormUgamma(bnorm_ugamma);
  model->SetCentralizeG(!utransform_G);
  model->SetUnitaryTransformG(utransform_G);
  openblas_set_num_threads(threads);
  if(fitnmin == ""){ if(siin == "") model->Solve(10000, 2000); }
  else model->ReadNullModelHDF5(fitnmin);
  if(siin == "") model->ComputeScoreInfo();
  else model->ReadScoreInfoHDF5(siin);

  PMatrix<double>* G = new PMatrix<double>("G", T->GetNRows(),
					   T->GetNCols(), false);
  PMatrix<double>* G_all = new PMatrix<double>("G_all", varfile,
					       vector<size_t>(),
					       vector<size_t>({0, 1}));
  G_all->PermuteRows(permutation);
  limitvar = std::max(nvariants, G_all->GetNCols());
  vector<size_t> rndm_index(limitvar);
  iota(rndm_index.begin(), rndm_index.end(), 0);
  random_device rd;
  mt19937 rgen(rd());
  shuffle(rndm_index.begin(), rndm_index.end(), rgen);
  
  vector<double> pval, pval_cent, score_stat, score_stat_cent;
  vector<double> Ugamma, Ugamma_cent, Ugamma_var, maf, mac;
#ifdef __CUDA__
  cublasHandle_t chandle;
  cublasCreate_v2(&chandle);
  model->SetCublasHandle(chandle);
#endif
  size_t ivar = 0;
  while(ivar < limitvar){
    G_all->GetBlock(0, rndm_index[ivar], G);
    model->SetG(G);
    model->ComputeScoreTest((ivar%10==0) ? true : false);
    score_stat.push_back(model->GetScoreStat());
    score_stat_cent.push_back(model->GetScoreStatCent());
    Ugamma.push_back(model->GetUgamma());
    Ugamma_cent.push_back(model->GetUgammaCent());
    Ugamma_var.push_back(model->GetUgammaVar());
    pval.push_back(model->GetPVal());
    pval_cent.push_back(model->GetPValCent());
    maf.push_back(model->GetMAF());
    mac.push_back(model->GetMAC());
    ivar ++;
    if(model->GetVarRatioConverged()) break;
  }
  if(ivar == limitvar){
    const vector<double>& vr = model->GetVarRatioEstimates();
    double mean = accumulate(vr.begin(), vr.end(), 0.0) / vr.size();
    double stddev = 0.0;
    for(const auto& r : vr) stddev += (r-mean) * (r-mean);
    stddev = sqrt(stddev / vr.size());
    cout << "r-hat convergence failed, current estimate: " << mean << " +/- "
	 << stddev << ", CV/neval=" << stddev/mean/vr.size() << endl;
    model->SetVarRatio(mean);
  }
  GHAWriteBGENScoreTestHDF5(model, testout, NULL, pval, pval_cent,
			    score_stat, score_stat_cent,
			    Ugamma, Ugamma_cent, Ugamma_var, maf, mac,
			    model->GetVarRatio());

  delete G_all;
  delete model;
  delete T;
  delete delta;
  delete X;
  delete psi;

  return 0;

}

#include "PMatrix.hh"
#include "BDMatrix.hh"
#include "GHAModel.hh"
#include "GHAIO.hh"
#include "SparseDosageReader.hh"
#include <H5Cpp.h>
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#include <cblas.h>
#include <vector>
#include <algorithm>
#ifdef __CUDA__
#include <cublas_v2.h>
#endif

using namespace std;
using namespace std::chrono;

double median(std::vector<double>& v) {
  size_t n = v.size();
  size_t mid = n / 2;
  
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double med = v[mid];
  
  if (n % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
    med = 0.5 * (med + v[mid - 1]);
  }
  
  return med;
}

int main(int argc, char* argv[]){

  string ghpath            = "../grmfit/gh/data";
  string ghbase            = "hermite";
  size_t ghdim             = 2;
  size_t ghorder           = 12;
  bool   ghstep            = false;
  size_t threads           = 4;
  size_t limitvar          = SIZE_MAX;
  bool limitvar_rhat       = false;
  size_t t_bins            = 0;
  string fitnmout          = "";
  string fitnmin           = "";
  string siout             = "";
  string siin              = "";
  string testout           = "";
  string cvfile            = "";
  size_t ncovariates       =  1;
  bool renorm_cv           = false;
  string grmfile           = "";
  bool grm_bd              = false;
  string varfile           = "";
  size_t nvariants         = 0;
  bool use_vr              = false;
  string svar_ratio        = "";
  double var_ratio         = 0.0;
  double epsilon_vr        = 1.0e-4;
  double spa_thresh        = numeric_limits<double>::max();
  bool cache_SI            = true;
  bool bnorm_ugamma        = false;
  bool utransform_G        = false;
  bool verbose             = false;
  const bool   b_last      = true;
  const bool   b_fullnr    = false;
  double epsilon           = 1.0e-5;
  bool absolute_thresh     = false;
  static struct option opts[]{
    {"help",             no_argument, NULL, 'h'},
    {"ghpath",     required_argument, NULL, 'g'},
    {"ghbase",     required_argument, NULL, 'G'},
    {"ndim",       required_argument, NULL, 'n'},
    {"order",      required_argument, NULL, 'o'},
    {"orderstep",        no_argument, NULL, 'O'},
    {"jthreads",   required_argument, NULL, 'j'},
    {"limitvar",   required_argument, NULL, 'l'},
    {"limitrhat",        no_argument, NULL, 'L'},
    {"tbins",      required_argument, NULL, 'T'},
    {"fitnmout",   required_argument, NULL, 'f'},
    {"fitnmin",    required_argument, NULL, 'F'},
    {"siout",      required_argument, NULL, 's'},
    {"siin",       required_argument, NULL, 'S'},
    {"testout",    required_argument, NULL, 't'},
    {"cvfile",     required_argument, NULL, 'c'},
    {"ncovariates",required_argument, NULL, 'C'},
    {"adjustcv",         no_argument, NULL, 'a'},
    {"absolutethresh",   no_argument, NULL, 'A'},
    {"grmfile",    required_argument, NULL, 'm'},
    {"grmbd",            no_argument, NULL, 'M'},
    {"varfile",    required_argument, NULL, 'b'},
    {"nvariants",  required_argument, NULL, 'B'},
    {"ratiovar",         no_argument, NULL, 'r'},
    {"ratioval",   required_argument, NULL, 'R'},
    {"epsilon",    required_argument, NULL, 'e'},
    {"epsilonvr",  required_argument, NULL, 'E'},
    {"spathresh",  required_argument, NULL, 'p'},
    {"donotcache",       no_argument, NULL, 'd'},
    {"ugammabnorm",      no_argument, NULL, 'u'},
    {"utransformG",      no_argument, NULL, 'U'},
    {"verbose",          no_argument, NULL, 'v'}};
  int opt = getopt_long(argc, argv,
			"hg:G:n:o:Oj:l:LT:f:F:s:S:t:c:C:aAm:Mb:B:rR:e:E:p:duUv",
			opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                              << endl
	   << "  -g path to GH data"                         << endl
	   << "  -n dimension of GH approx"                  << endl
	   << "  -o order     of GH approx"                  << endl
	   << "  -O step GH order"                           << endl
	   << "  -j number of threads"                       << endl
	   << "  -l limit number variants to read"           << endl
	   << "  -L stop reading variants once r-hat found"  << endl
	   << "  -T parameter for rebinning times"           << endl
	   << "  -f null model fit output"                   << endl
	   << "  -F null model fit input"                    << endl
	   << "  -s score/information output"                << endl
	   << "  -S score/information input"                 << endl
	   << "  -t score test output"                       << endl
	   << "  -c covariate csv file"                      << endl
	   << "  -C number of covariates"                    << endl
	   << "  -a adjust mean/sigma of covariates"         << endl
	   << "  -m grm hdf5 file"                    << endl
	   << "  -M grm block diagonal"               << endl
	   << "  -b bgen to hdf5 variant file"               << endl
	   << "  -B number of variants in csv file"          << endl
	   << "  -r use variance ratio for scores"           << endl
	   << "  -R set variance ratio (either val or file)" << endl
	   << "  -e set epsilon for null model"              << endl
	   << "  -E epsilon for variance ratio"              << endl
    << "  -p use SPA when abs(score) >= value"         << endl
	   << "  -d do not cache Stheta and Igamma"          << endl
	   << "  -v verbose mode"                            << endl;
      return 0;
    case 'g': ghpath      = string(optarg); break;
    case 'G': ghbase      = string(optarg); break;
    case 'n': ghdim       =   atoi(optarg); break;
    case 'o': ghorder     =   atoi(optarg); break;
    case 'O': ghstep      =true; ghorder=4; break;
    case 'j': threads     =   atoi(optarg); break;
    case 'l': limitvar    =   atoi(optarg); break;
    case 'L': limitvar_rhat=          true; break;
    case 'T': t_bins      =   atoi(optarg); break;
    case 'f': fitnmout    = string(optarg); break;
    case 'F': fitnmin     = string(optarg); break;
    case 's': siout       = string(optarg); break;
    case 'S': siin        = string(optarg); break;
    case 't': testout     = string(optarg); break;
    case 'c': cvfile      = string(optarg); break;
    case 'C': ncovariates =   atoi(optarg); break;
    case 'a': renorm_cv   =           true; break;
    case 'A': absolute_thresh =       true; break;
    case 'm': grmfile     = string(optarg); break;
    case 'M': grm_bd      =           true; break;
    case 'b': varfile     = string(optarg); break;
    case 'B': nvariants   =   atoi(optarg); break;
    case 'r': use_vr      =           true; break;
    case 'R': svar_ratio  = string(optarg); break;
    case 'e': epsilon     =   stod(optarg); break;
    case 'E': epsilon_vr  =   stod(optarg); break;
    case 'p': spa_thresh  =   stod(optarg); break;
    case 'd': cache_SI    =          false; break;
    case 'u': bnorm_ugamma=           true; break;
    case 'U': utransform_G=           true; break;
    case 'v': verbose     =           true; break;
    default:
      cout << "unrecoginized option -" << string(""+opt) << endl;
      return 1;
    }
    opt = getopt_long(argc, argv,
		      "hg:G:n:o:Oj:l:LT:f:F:s:S:t:c:C:aAm:Mb:B:rR:e:E:p:duUv",
		      opts, NULL);
  }

  assert(cvfile  != "");
  assert(grmfile != "");
  assert(fitnmin == "" || fitnmout == "");
  assert(siin    == "" || siout    == "");

  // set rhat if provided either as value or from file
  if(svar_ratio.size() > 0){
    char* end_test;
    var_ratio = strtod(svar_ratio.c_str(), &end_test);
    if(end_test == svar_ratio.c_str() || *end_test != '\0'){
      H5::H5File* fvr = new H5::H5File(svar_ratio.c_str(), H5F_ACC_RDONLY);
      ReadHDF5scalar<double>(fvr, var_ratio, "var_ratio");
      delete fvr;
    }
  }

  cout << "reading input data..." << endl;
  auto t0 = high_resolution_clock::now();
  PMatrix<double>*     T = new PMatrix<double>("T", cvfile,
					       vector<size_t>({2}));
  if(t_bins > 0){
    size_t nbins = T->BinValues(t_bins);
    cout << "condensed observation times into " << nbins
	 << " variable width bins" << endl;
  }
  PMatrix<double>* delta = new PMatrix<double>("delta", cvfile,
					       vector<size_t>({1}));
  vector<size_t> cvindex(ncovariates);
  iota(cvindex.begin(), cvindex.end(), 3);
  PMatrix<double>*     X = new PMatrix<double>("X", cvfile, cvindex);
  if(renorm_cv){
    for(size_t i=0; i<X->GetNCols(); i++){
      bool bincol = true;
      for(size_t j=0; j<X->GetNRows(); j++){
	double x = X->GetVal(j, i);
	if(x != (double) ((int) x) || x < 0.0 || x > 1.0){
	  bincol = false;
	  break;
	}
      }
      if(!bincol) X->RenormalizeCol(i);
    }
  }
  H5::H5File grmh5(grmfile.c_str(), H5F_ACC_RDONLY);
  BDMatrix<double>*  psi;
  if(!grm_bd){
    psi = new BDMatrix<double>("psi");
    PMatrix<double>* grm = new PMatrix<double>("sGRM", grmh5);
    grmh5.close();
    psi->AppendBlock(grm);
    delete grm;
    cout << "Appended single dense block to GRM with " << psi->GetNBlocks()
	 << " blocks after diagonalization" << endl;
  }
  else{
    psi = new BDMatrix<double>("psi", "block_", grmh5, true, false);
    grmh5.close();
  }
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
  model->SetAbsoluteThresh(absolute_thresh);
  model->SetBLast(b_last);
  model->SetBFullNR(b_fullnr);
  model->SetVerbose(verbose);
  model->SetGHStepIterConv(50);
  model->SetUseVarRatio(use_vr);
  if(var_ratio > 0.0) model->SetVarRatio(var_ratio);
  model->SetVarRatioEpsilon(epsilon_vr);
  model->SetSPAThreshold(spa_thresh);
  model->SetCacheSthetaIgamma(cache_SI);
  model->SetBNormUgamma(bnorm_ugamma);
  model->SetCentralizeG(!utransform_G);
  model->SetUnitaryTransformG(utransform_G);
  openblas_set_num_threads(threads);
  #ifdef __CUDA__
    cublasHandle_t chandle;
    cublasCreate_v2(&chandle);
    model->SetCublasHandle(chandle);
  #endif
  if(fitnmin == ""){
    if(siin == "") model->Solve(100000, 2000);
    if(fitnmout != "") model->WriteNullModelHDF5(fitnmout);
  }
  else model->ReadNullModelHDF5(fitnmin);
  if(siin == ""){
    if(siout != ""){
      model->ComputeScoreInfo();
      model->WriteScoreInfoHDF5(siout);
    }
    else if(testout != ""){
      if(fitnmin == "" && siin == "") model->Solve(100000, 2000);
      model->ComputeScoreInfo();
    }
  }
  else model->ReadScoreInfoHDF5(siin);
  if(testout != "" || nvariants > 0){
    PMatrix<double>* G = new PMatrix<double>("G", T->GetNRows(),
					     T->GetNCols(), false);
    PMatrix<double>* G_all = NULL;
    SparseDosageReader* reader = NULL;
    vector<double> pval;
    if(varfile != ""){
      if(testout != ""){
	reader = new SparseDosageReader(varfile);
	if(T->GetNRows() < reader->GetNInd()){
	  permutation.resize(reader->GetNInd());
	  iota(permutation.begin()+psi->GetNRows(),
	       permutation.end(), psi->GetNRows());
	  G_all = new PMatrix<double>("G_all", reader->GetNInd(),
				      T->GetNCols(), false);
	}
	delete reader;
	reader = new SparseDosageReader(varfile, permutation);
	pval.resize(std::min(reader->GetNVariants(), limitvar));
      }
    }
    else{
      vector<size_t> gindex(nvariants);
      iota(gindex.begin(), gindex.end(), 3+ncovariates);
      G_all = new PMatrix<double>("G_all", cvfile, gindex);
      G_all->PermuteRows(permutation);
      pval.resize(nvariants);
      limitvar = nvariants;
    }
    vector<double> pval_gc(pval.size());
    vector<double> score_stat(pval.size());
    vector<double> Ugamma(pval.size());
    vector<double> Ugamma_var(pval.size());
    vector<double> maf(pval.size());
    vector<double> mac(pval.size());
    size_t ivar = 0;
    while((reader ? reader->NextVariant(G_all ? G_all : G) : true) &&
	  ivar < limitvar){
      if(reader){ if(G_all) G_all->GetBlock(0, 0, G); }
      else G_all->GetBlock(0, ivar, G);
      model->SetG(G);
      model->ComputeScoreTest((ivar%1000==0) ? true : false);
      score_stat[ivar]      = model->GetScoreStat();
      Ugamma[ivar]          = model->GetUgamma();
      Ugamma_var[ivar]      = model->GetUgammaVar();
      pval[ivar]            = model->GetPVal();
      maf[ivar]             = model->GetMAF();
      mac[ivar]             = model->GetMAC();
      ivar ++;
      if(ivar % 1000 == 0 && reader) reader->PrintTime();
      if(limitvar_rhat && model->GetVarRatioConverged()) break;
    }
    GHAWriteBGENScoreTestHDF5(model, testout, reader, pval,
                                score_stat,
                                Ugamma, Ugamma_var, maf, mac,
                                model->GetVarRatio());
    
    if(reader) delete reader;
    if(G_all)  delete G_all;
  }

  delete model;
  delete T;
  delete delta;
  delete X;
  delete psi;

  return 0;

}

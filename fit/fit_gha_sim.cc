#include "PMatrix.hh"
#include "BDMatrix.hh"
#include "GHAModel.hh"
#include "GHAIO.hh"
#include "SparseDosageReader.hh"
#include <Math/GSLRndmEngines.h>
#include <TRandom.h>
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#ifdef __CUDA__
#include <cublas_v2.h>
#endif

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

  size_t nthreads = 4;
  vector<size_t> brows;
  vector<size_t> bcount;
  string ghpath      = "../grmfit/gh/data";
  string ghbase      = "hermite";
  size_t ghdim       = 2;
  size_t ghorder     = 12;
  bool   ghstep      = false;
  bool   verbose     = false;
  double tausim      = 0.25;
  double lambdasim   = 0.3;
  vector<double> betasim;
  double epsilon     = 1.0e-5;
  size_t limitvar    = SIZE_MAX;
  string fitnmout    = "";
  string fitnmin     = "";
  string siout       = "";
  string siin        = "";
  string testout     = "";
  string cvfile      = "";
  size_t ncovariates =  1;
  string varfile     = "";
  size_t nvariants   = 0;
  bool use_vr        = false;
  double var_ratio   = 0.0;
  double epsilon_vr  = 1.0e-4;
  double spa_thresh  = numeric_limits<double>::max();
  bool cache_SI      = true;
  bool bnorm_ugamma  = false;
  bool utransform_G  = false;
  size_t skip_rndm   = 0;
  static struct option opts[]{
    {"help",             no_argument, NULL, 'h'},
    {"threads",    required_argument, NULL, 'j'},
    {"brows",      required_argument, NULL, 'x'},
    {"bcount",     required_argument, NULL, 'X'},
    {"ghpath",     required_argument, NULL, 'g'},
    {"ghbase",     required_argument, NULL, 'G'},
    {"ndim",       required_argument, NULL, 'n'},
    {"order",      required_argument, NULL, 'o'},
    {"orderstep",        no_argument, NULL, 'O'},
    {"verbose",          no_argument, NULL, 'v'},
    {"tau",        required_argument, NULL, 'T'},
    {"lambda",     required_argument, NULL, 'L'},
    {"epsilon",    required_argument, NULL, 'e'},
    {"limitvar",   required_argument, NULL, 'l'},
    {"beta",       required_argument, NULL, 'P'},
    {"fitnmout",   required_argument, NULL, 'f'},
    {"fitnmin",    required_argument, NULL, 'F'},
    {"siout",      required_argument, NULL, 's'},
    {"siin",       required_argument, NULL, 'S'},
    {"testout",    required_argument, NULL, 't'},
    {"cvfile",     required_argument, NULL, 'c'},
    {"ncovariates",required_argument, NULL, 'C'},
    {"varfile",    required_argument, NULL, 'b'},
    {"nvariants",  required_argument, NULL, 'B'},
    {"ratiovar",         no_argument, NULL, 'r'},
    {"ratioval",   required_argument, NULL, 'R'},
    {"epsilonvr",  required_argument, NULL, 'E'},
    {"spathresh",  required_argument, NULL, 'p'},
    {"donotcache",       no_argument, NULL, 'd'},
    {"ugammabnorm",      no_argument, NULL, 'u'},
    {"utransformG",      no_argument, NULL, 'U'},
    {"skiprndm",   required_argument, NULL, 'k'}};
  int opt = getopt_long(argc, argv,
			"hj:x:X:g:G:n:o:Ov:T:L:e:l:P:f:F:s:S:t:c:C:b:B:rR:E:p:duUk:",
			opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                            << endl
	   << "  -t number of threads"               << endl
	   << "  -x append blocks of size"           << endl
	   << "  -X append number of blocks"         << endl
	   << "  -g path to GH data"                 << endl
	   << "  -G base of GH files"                << endl
	   << "  -n dimension of GH approx"          << endl
	   << "  -o order     of GH approx"          << endl
	   << "  -O step GH order"                   << endl
	   << "  -v verbose mode"                    << endl
	   << "  -T simulated tau"                   << endl
	   << "  -L simulated lambda"                << endl
	   << "  -e convergence level"               << endl
	   << "  -l limit number variants to read"   << endl
	   << "  -P append value for beta"           << endl
	   << "  -f null model fit output"           << endl
	   << "  -F null model fit input"            << endl
	   << "  -s score/information output"        << endl
	   << "  -S score/information input"         << endl
	   << "  -t score test output"               << endl
	   << "  -c covariate txt file"              << endl
	   << "  -C number of covariates"            << endl
	   << "  -b bgen to hdf5 variant file"       << endl
	   << "  -B number of variants in csv file"  << endl
	   << "  -r use variance ratio for scores"   << endl
	   << "  -R set the variance ratio"          << endl
	   << "  -E epsilon for variance ratio"      << endl
	   << "  -p use SPA when abs(score) < value" << endl
	   << "  -d do not cache Stheta and Igamma"  << endl
	   << "  -u normalize Ugamma by block size"  << endl
	   << "  -U apply unitary transform to G"    << endl
	   << "  -k mean number of variants to skip" << endl;
      return 0;
    case 'j': nthreads     =    atoi(optarg);   break;
    case 'x': brows.push_back(  atoi(optarg));  break;
    case 'X': bcount.push_back( atoi(optarg));  break;
    case 'g': ghpath       =  string(optarg);   break;
    case 'G': ghbase       =  string(optarg);   break;
    case 'n': ghdim        =    atoi(optarg);   break;
    case 'o': ghorder      =    atoi(optarg);   break;
    case 'O': ghstep       =            true;   break;
    case 'v': verbose      =            true;   break;
    case 'T': tausim       =    stod(optarg);   break;
    case 'L': lambdasim    =    stod(optarg);   break;
    case 'e': epsilon      =    stod(optarg);   break;
    case 'P': betasim.push_back(stod(optarg));  break;
    case 'l': limitvar     =    atoi(optarg);   break;
    case 'f': fitnmout     =  string(optarg);   break;
    case 'F': fitnmin      =  string(optarg);   break;
    case 's': siout        =  string(optarg);   break;
    case 'S': siin         =  string(optarg);   break;
    case 't': testout      =  string(optarg);   break;
    case 'c': cvfile       =  string(optarg);   break;
    case 'C': ncovariates  =    atoi(optarg);   break;
    case 'b': varfile      =  string(optarg);   break;
    case 'B': nvariants    =    atoi(optarg);   break;
    case 'r': use_vr       =            true;   break;
    case 'R': var_ratio    =    stod(optarg);   break;
    case 'E': epsilon_vr   =    stod(optarg);   break;
    case 'p': spa_thresh   =    stod(optarg);   break;
    case 'd': cache_SI     =           false;   break;
    case 'u': bnorm_ugamma =            true;   break;
    case 'U': utransform_G =            true;   break;
    case 'k': skip_rndm    =    atoi(optarg);   break;
    default:
      cout << "unrecoginized option -" << string(""+opt) << endl;
      return 1;
    }
    opt = getopt_long(argc, argv,
		      "hj:x:X:g:G:n:o:Ov:T:L:e:l:P:f:F:s:S:t:c:C:b:B:rR:E:p:duUk:",
		      opts, NULL);
  }
  if(ghpath == ""){
    cout << "must specify path to GH files with -g" << endl;
  }
  assert(tausim > 0.0);
  assert(epsilon > 0.0 && epsilon < 1.0);
  assert(betasim.size() != 0);
  assert(ghorder < 129);
  assert(cvfile  != "");
  assert(fitnmin == "" || fitnmout == "");
  assert(siin    == "" || siout    == "");
  const size_t npar = betasim.size();
  openblas_set_num_threads(nthreads);
  
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
  PMatrix<double>* block10 = new PMatrix<double>("block10", 10, 10, false);
  block10->SetData(vblock);
  
  assert(brows.size() > 0 && brows.size() == bcount.size());
  BDMatrix<double>* psi = new BDMatrix<double>("psi");
  for(size_t i=0; i<brows.size(); i++){
    assert(brows[i] <= 10);
    PMatrix<double>* block = new PMatrix<double>("block_"+to_string(brows[i]),
						 brows[i], brows[i], false);
    block10->GetBlock(block10->GetNRows()-brows[i],
		      block10->GetNCols()-brows[i], block);
    for(size_t j=0; j<bcount[i]; j++) psi->AppendBlock(block);
    delete block;
  }
  delete block10;
  
  cout << "sorting grm by block size..." << endl;
  auto t0 = high_resolution_clock::now();
  psi->SortBlocks();
  vector<size_t> permutation = psi->GetIndexPermutation();
  auto t1 = high_resolution_clock::now();
  cout << "  " <<
    duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;

  gRandom->SetSeed();
  const int seed = gRandom->GetSeed();
  ROOT::Math::GSLRandomEngine gsl_rand;
  gsl_rand.Initialize();
  gsl_rand.SetSeed(seed);

  PMatrix<double>* T = NULL, *delta = NULL, *X = NULL;
  if(fitnmin == "" && siin == ""){
    BDMatrix<double>* taupsi = new BDMatrix<double>("taupsi", *psi);
    (*taupsi) *= tausim;

    T     = new PMatrix<double>("T",     psi->GetNRows(),    1, false);
    delta = new PMatrix<double>("delta", psi->GetNRows(),    1, false);
    X     = new PMatrix<double>("X",     psi->GetNRows(), npar, false);
    
    cout << "generating simulated dataset..." << endl;
    t0 = high_resolution_clock::now();
    size_t offset = 0;
    for(size_t i=0; i<taupsi->GetNBlocks(); i++){
      const size_t nrows = taupsi->GetBlockRows(i);
      vector<double> m(nrows, 0.0), b(nrows, 0.0);
      gsl_rand.GaussianND(nrows, &m.front(),
			  &taupsi->GetData().front() +
			  taupsi->GetBlockDataOffset(i), &b.front());
      size_t k = offset;
      for(size_t j=0; j<nrows; (j++, k++)){
	X->SetVal(k, 0, gsl_rand()<=0.5 ? 0.0 : 1.0);
	X->SetVal(k, 1, gsl_rand.Gaussian(1.0));
	double c = std::min(3.0, 4*gsl_rand());
	double t = -log(gsl_rand()) / lambdasim;
	double arg = b[j];
	for(size_t xi=0; xi<npar; xi++) arg += X->GetVal(k, xi) * betasim[xi];
	t /= exp(arg);
	T->SetVal(k, std::min(c, t));
	delta->SetVal(k, c >= t ? 1.0 : 0.0);
      }
      offset += nrows;
    }
    t1 = high_resolution_clock::now();
    cout << "  " <<
      duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;
    delete taupsi;
    H5File* h5 = new H5File(cvfile.c_str(), H5F_ACC_TRUNC);
    T->WriteHDF5(*h5, "T");
    delta->WriteHDF5(*h5, "delta");
    X->WriteHDF5(*h5, "X");
    delete h5;
  }
  else{
    H5File* h5 = new H5File(cvfile.c_str(), H5F_ACC_RDONLY);
    T     = new PMatrix<double>("T",     *h5);
    delta = new PMatrix<double>("delta", *h5);
    X     = new PMatrix<double>("X",     *h5);
    delete h5;
  }
      
  GHAModel* model = new GHAModel(ghpath, ghbase, ghdim, ghorder, ghstep,
				 T, delta, X, psi);
  model->SetVerbose(verbose);
  model->SetEpsilon(epsilon);
  model->SetBFullNR(false);
  model->SetBLast(true);
  model->SetGHStepIterConv(50);
  model->SetUseVarRatio(use_vr);
  if(var_ratio > 0.0) model->SetVarRatio(var_ratio);
  model->SetVarRatioEpsilon(epsilon_vr);
  model->SetSPAThreshold(spa_thresh);
  model->SetCacheSthetaIgamma(cache_SI);
  model->SetBNormUgamma(bnorm_ugamma);
  model->SetCentralizeG(!utransform_G);
  model->SetUnitaryTransformG(utransform_G);
  if(fitnmin == ""){
    if(siin == "") model->Solve(10000, 2000);
    if(fitnmout != "") model->WriteNullModelHDF5(fitnmout);
  }
  else model->ReadNullModelHDF5(fitnmin);
  if(siin == ""){
    if(siout != ""){
      model->ComputeScoreInfo();
      model->WriteScoreInfoHDF5(siout);
    }
    else if(testout != ""){
      if(fitnmin == "" && siin == "") model->Solve(10000, 1000);
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
      }
    }
    else{
      vector<size_t> gindex(nvariants);
      iota(gindex.begin(), gindex.end(), 3+ncovariates);
      G_all = new PMatrix<double>("G_all", cvfile, gindex);
      G_all->PermuteRows(permutation);
      limitvar = nvariants;
    }
    vector<double> pval_cent;
    vector<double> score_stat;
    vector<double> score_stat_cent;
    vector<double> Ugamma;
    vector<double> Ugamma_cent;
    vector<double> Ugamma_var;
    vector<double> maf;
    vector<double> mac;
#ifdef __CUDA__
    cublasHandle_t chandle;
    cublasCreate_v2(&chandle);
    model->SetCublasHandle(chandle);
#endif
    size_t ivar = 0, jvar = 0, nskip = 0;
    while((reader ? reader->NextVariant(G_all ? G_all : G) : true) &&
	  ivar < limitvar){
      if(reader){ if(G_all) G_all->GetBlock(0, 0, G); }
      else G_all->GetBlock(0, ivar, G);
      if(nskip > 0){
	nskip --;
	ivar ++;
	if(ivar % 1000 == 0 && reader) reader->PrintTime();
	continue;
      }
      model->SetG(G);
      model->ComputeScoreTest((jvar%100==0) ? true : false);
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
      jvar ++;
      if(ivar % 1000 == 0 && reader) reader->PrintTime();
      if(skip_rndm > 0) nskip = (int) (2 * skip_rndm * gsl_rand());
    }
    GHAWriteBGENScoreTestHDF5(model, testout, reader, pval, pval_cent,
			      score_stat, score_stat_cent,
			      Ugamma, Ugamma_cent, Ugamma_var, maf, mac,
			      model->GetVarRatio());
    if(reader) delete reader;
    if(G_all)  delete G_all;
  }

  delete T;
  delete delta;
  delete X;
  delete psi;
  delete model;
  
  return 0;
}

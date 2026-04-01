#include "PMatrix.hh"
#include "GRMModel.hh"
#include <Math/GSLRndmEngines.h>
#include <TH1.h>
#include <TFile.h>
#include <TRandom.h>
#include <TParameter.h>
#include <TVectorT.h>
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#include <cblas-openblas.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

  size_t nds      = 1;
  size_t dsstart  = 0;
  size_t nthreads = 6;
  size_t bsize    = 2;
  size_t nblock   = 1000;
  size_t ngrid    = 100;
  size_t ghorder  = 20;
  int seed        = -1;
  bool verbose    = false;
  bool b_full     = false;
  bool b_zero     = false;
  size_t biter    = 1;
  double relcoeff = 0.5;
  double tausim   = 1.0;
  double taumax   = 2.0;
  double lambda   = 0.3;
  double epsilon  = 1.0e-5;
  string outdir   = ".";
  string ghpath   = "/home/tcald/bios/hermite/data";
  vector<double> betasim;
  static struct option opts[]{
    {"help",           no_argument, NULL, 'h'},
    {"nds",      required_argument, NULL, 'n'},
    {"ndsstart", required_argument, NULL, 'N'},
    {"nthreads", required_argument, NULL, 'j'},
    {"dimblock", required_argument, NULL, 'd'},
    {"blocks",   required_argument, NULL, 'b'},
    {"gridn",    required_argument, NULL, 'g'},
    {"ghorder",  required_argument, NULL, 'G'},
    {"seed",     required_argument, NULL, 's'},
    {"fullb",          no_argument, NULL, 'f'},
    {"zerob",          no_argument, NULL, 'z'},
    {"maxbiter", required_argument, NULL, 'm'},
    {"relcoeff", required_argument, NULL, 'r'},
    {"tau",      required_argument, NULL, 't'},
    {"taumax",   required_argument, NULL, 'T'},
    {"lambda",   required_argument, NULL, 'l'},
    {"beta",     required_argument, NULL, 'B'},
    {"epsilon",  required_argument, NULL, 'e'},
    {"pathgh",   required_argument, NULL, 'p'},
    {"outdir",   required_argument, NULL, 'o'}};
  int opt = getopt_long(argc, argv,
			 "hn:N:j:d:b:g:G:s:vfzm:r:t:T:l:B:e:p:o:",
			opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options: "                << endl
	   << "  -n number of datasets"  << endl
	   << "  -N ds start index"      << endl
	   << "  -j number of threads"   << endl
	   << "  -d dimension of block"  << endl
	   << "  -b number of blocks"    << endl
	   << "  -g grid points in tau"  << endl
	   << "  -G gauss-hermite order" << endl
	   << "  -s rng seed"            << endl
	   << "  -v verbose"             << endl
	   << "  -f full laplace for b"  << endl
	   << "  -z zero b at each iter" << endl
	   << "  -m max NR iter for b"   << endl
	   << "  -r relation coeff"      << endl
	   << "  -t simulated tau"       << endl
	   << "  -T tau maximum in GS"   << endl
	   << "  -l simulated lambda"    << endl
	   << "  -B append beta"         << endl
	   << "  -e epsilon convergence" << endl
	   << "  -o output directory"    << endl;
      break;
    case 'n': nds      = atoi(optarg);         break;
    case 'N': dsstart  = atoi(optarg);         break;
    case 'j': nthreads = atoi(optarg);         break;
    case 'd': bsize    = atoi(optarg);         break;
    case 'b': nblock   = atoi(optarg);         break;
    case 'g': ngrid    = atoi(optarg);         break;
    case 'G': ghorder  = atoi(optarg);         break;
    case 's': seed     = atoi(optarg);         break;
    case 'v': verbose  = true;                 break;
    case 'f': b_full   = true;                 break;
    case 'z': b_zero   = true;                 break;
    case 'm': biter    = atoi(optarg);         break;
    case 'r': relcoeff = atof(optarg);         break;
    case 't': tausim   = atof(optarg);         break;
    case 'T': taumax   = atof(optarg);         break;
    case 'l': lambda   = atof(optarg);         break;
    case 'e': epsilon  = stod(optarg);         break;
    case 'p': ghpath   = string(optarg);       break;
    case 'o': outdir   = string(optarg);       break;
    case 'B': betasim.push_back(atof(optarg)); break;
    default:
      cout << "skipping unrecognized option " << "-" << opt << endl;
      break;
    }
    opt = getopt_long(argc, argv,
		       "hn:N:j:d:b:g:G:s:vfzm:r:t:T:l:B:e:p:o:",opts,NULL);
  }
  assert(bsize >= 1 && nblock >= 1);
  assert(biter > 0);
  assert(relcoeff >= 0.0 && relcoeff < 1.0);
  assert(tausim > 0.0 && taumax > 0 && ngrid > 0 && taumax > tausim);
  assert(epsilon > 0.0 && epsilon < 1.0);
  assert(betasim.size() == 2);
  assert(ghorder < 129);
  if(outdir[outdir.size()-1] != '/') outdir += '/';

  const size_t nunrel = (bsize == 10) ? 5000 : 0;
  if(bsize == 10) nblock = 500;
  const size_t npar = betasim.size();
  const size_t nind = nblock * bsize + nunrel;

  bool user_seed = true;
  if(seed < 0){
    gRandom->SetSeed();
    seed = gRandom->GetSeed();
    user_seed = false;
  }
  else gRandom->SetSeed(seed);
  ROOT::Math::GSLRandomEngine gsl_rand;
  gsl_rand.Initialize();
  gsl_rand.SetSeed(seed);

  PMatrix<double>*   psi = new PMatrix<double>("psi", nind, nind, true);
  const size_t nindbl = nblock * bsize;
  vector<size_t> bsizes;
  for(size_t i=0; i<nindbl; i+=bsize){
    bsizes.push_back(bsize);
    if(bsize == 10){
      for(size_t j=0; j<2; j++){
	psi->SetVal(i+j, i+2, 0.5);  psi->SetVal(i+j, i+3, 0.5);
	psi->SetVal(i+j, i+6, 0.25); psi->SetVal(i+j, i+7, 0.25);
	psi->SetVal(i+j, i+8, 0.25); psi->SetVal(i+j, i+9, 0.25);
      }
      psi->SetVal(i+2, i+3, 0.5);   psi->SetVal(i+2, i+6, 0.5);
      psi->SetVal(i+2, i+7, 0.5);   psi->SetVal(i+2, i+8, 0.25);
      psi->SetVal(i+2, i+9, 0.25);  psi->SetVal(i+3, i+6, 0.25);
      psi->SetVal(i+3, i+7, 0.25);  psi->SetVal(i+3, i+8, 0.5);
      psi->SetVal(i+3, i+9, 0.5);   psi->SetVal(i+4, i+6, 0.5);
      psi->SetVal(i+4, i+7, 0.5);   psi->SetVal(i+5, i+8, 0.5);
      psi->SetVal(i+5, i+9, 0.5);   psi->SetVal(i+6, i+7, 0.5);
      psi->SetVal(i+6, i+8, 0.125); psi->SetVal(i+6, i+9, 0.125);
      psi->SetVal(i+7, i+8, 0.125); psi->SetVal(i+7, i+9, 0.125);
      psi->SetVal(i+8, i+9, 0.5);
      for(size_t j=0; j<bsize; j++)
	for(size_t k=0; k<j; k++)
	  psi->SetVal(i+j, i+k, psi->GetVal(i+k, i+j));
    }
    else{
      for(size_t j=0; j<bsize; j++)
	for(size_t k=0; k<bsize; k++)
	  if(j != k) psi->SetVal(i+j, i+k, relcoeff);
      if(bsize == 3){
	psi->SetVal(i, i+2, 0.25);
	psi->SetVal(i+2, i, 0.25);
      }
    }
  }
  for(size_t i=nindbl; i<nind; i++) bsizes.push_back(1);

  vector<pair<size_t, size_t> > sample(bsizes.size(), make_pair(0, 0));
  for(size_t i=0; i<sample.size(); i++){
    if(bsizes[i] < 2) continue;
    sample[i].first  = (size_t) (gsl_rand() *  bsizes[i]);
    sample[i].second = (size_t) (gsl_rand() * (bsizes[i]-1));
    if(sample[i].second == sample[i].first) sample[i].second = bsizes[i]-1;
  }
  
  for(size_t ids=0; ids<nds; ids++){
    cout << "ds " << ids << "..." << endl;
    PMatrix<double>*      T = new PMatrix<double>("T",     nind, 1,    false);
    PMatrix<double>*  delta = new PMatrix<double>("delta", nind, 1,    false);
    PMatrix<double>*      X = new PMatrix<double>("X",     nind, npar, false);
    PMatrix<double>* psitmp = new PMatrix<double>("psitmp", *psi);
    cout << "generating dataset...";
    auto t0 = high_resolution_clock::now();
    (*psitmp) *= tausim;
    vector<double> m(nind, 0.0), b(nind, 0.0);
    gsl_rand.GaussianND(nind, &m.front(),
			&psitmp->GetData().front(), &b.front());
    for(size_t i=0; i<nind; i++){
      X->SetVal(i, 0, gsl_rand()<=0.5 ? 0.0 : 1.0);
      X->SetVal(i, 1, gsl_rand.Gaussian(1.0));
      double c = std::min(3.0, 4*gsl_rand());
      double t = -log(gsl_rand()) / lambda;
      t /= exp(X->GetVal(i, 0)*betasim[0] + X->GetVal(i, 1)*betasim[1] + b[i]);
      T->SetVal(i, std::min(c, t));
      delta->SetVal(i, c >= t ? 1.0 : 0.0);
    }
    delete psitmp;
    auto t1 = high_resolution_clock::now();
    double time_s = duration_cast<nanoseconds>(t1-t0).count()/1.0e9;
    cout << scientific << setprecision(6) << time_s << " s" << endl;

    GRMModel* model = new GRMModel(T, delta, X, psi);
    model->SetEpsilon(epsilon);
    model->SetbPartialLaplace(!b_full);
    model->SetbFromLast(!b_zero);
    model->LoadGaussHermite(ghpath, "hermite", ghorder);
    model->SetVerbose(verbose);
    openblas_set_num_threads(nthreads);
    t0 = high_resolution_clock::now();
    //model->GridSearchTau(100, biter, 0.0, taumax, ngrid);
    //model->Solve(0, biter);
    model->SolveGH2(0, biter, bsizes, sample);
    t1 = high_resolution_clock::now();
    time_s = duration_cast<nanoseconds>(t1-t0).count()/1.0e9;
    cout << scientific << setprecision(6) << time_s << " s" << endl;
    vector<double> nll = model->GetTauNLL();
    vector<vector<double> > bnll = model->GetBetaNLL();
    double nllt = 0.0;//model->EvalNLL(tausim, betasim);
    cout << "nll @ sim = " << nllt << endl;
    delete model;
    delete T;
    delete delta;
    delete X;
    /*TH1D* hnll = new TH1D("hnll", "", ngrid, 0.0, taumax);
    vector<TH1D*> hbnll(npar, NULL);
    for(size_t i=0; i<npar; i++)
      hbnll[i] = new TH1D(("hbnll_"+to_string(i)).c_str(), "",
			  ngrid, 0.0, taumax);
    for(size_t i=0; i<ngrid; i++){
      hnll->SetBinContent(i+1, nll[i]);
      for(size_t j=0; j<npar; j++)
	hbnll[j]->SetBinContent(i+1, bnll[i][j]);
	}*/
    TH1D* hnll = new TH1D("hnll", "", nll.size(), 0.0, nll.size());
    vector<TH1D*> hbnll(npar, NULL);
    for(size_t i=0; i<npar; i++)
      hbnll[i] = new TH1D(("hbnll_"+to_string(i)).c_str(), "",
			  2000, 0.0, taumax);
    for(size_t i=0; i<nll.size(); i++){
      hnll->SetBinContent(i+1, nll[i]);
      for(size_t j=0; j<npar; j++)
	hbnll[j]->SetBinContent(hbnll[j]->GetXaxis()->FindBin(nll[i]),
				bnll[i][j]);
    }
    size_t jds = dsstart + ids;
    string ofname = "grm_sim_d" + to_string(bsize) + "_b" + to_string(nblock);
    ofname += "_r"  + to_string((int)(relcoeff*1000));
    ofname += "_gh" + to_string(ghorder) + "_t" + to_string((int)(tausim*100));
    ofname += "_l"  + to_string((int)(lambda*100));
    ofname += "_s"  + to_string(user_seed?seed:0);
    ofname += "_ds" + to_string(jds) + ".root";
    TFile* outfile = new TFile((outdir+ofname).c_str(),"recreate");
    outfile->WriteObject(hnll, hnll->GetName());
    for(size_t i=0; i<npar; i++)
      outfile->WriteObject(hbnll[i], hbnll[i]->GetName());
    TParameter<double>* rtime_s = new TParameter<double>("time_s",   time_s);
    TParameter<double>* rtausim = new TParameter<double>("tausim",   tausim);
    TParameter<double>* rtaumax = new TParameter<double>("taumax",   taumax);
    TParameter<double>* rlambda = new TParameter<double>("lambda",   lambda);
    TParameter<double>* rrelcoeff=new TParameter<double>("relcoeff", relcoeff);
    TParameter<double>* repsilon= new TParameter<double>("epsilon",  epsilon);
    TParameter<int>*    rngrid  = new TParameter<int>("ngrid",    ngrid);
    TParameter<int>*    rbsize  = new TParameter<int>("bsize",    bsize);
    TParameter<int>*    rnblock = new TParameter<int>("nblock",   nblock);
    TParameter<int>*    rghorder= new TParameter<int>("ghorder",  ghorder);
    TParameter<int>*    rseed   = new TParameter<int>("seed",     seed);
    TParameter<int>*    rnthreads=new TParameter<int>("nthreads", nthreads);
    TParameter<int>*    rbiter  = new TParameter<int>("biter",    biter);
    TParameter<int>*    rnunrel = new TParameter<int>("nunrel",   nunrel);
    TParameter<bool>*   rb_full = new TParameter<bool>("b_full",  b_full);
    TParameter<bool>*   rb_zero = new TParameter<bool>("b_zero",  b_zero);
    TVectorT<double>*   rbetasim= new TVectorT<double>(betasim.size(),
						       &betasim.front());
    outfile->WriteObject(rtime_s,   rtime_s->GetName());
    outfile->WriteObject(rtausim,   rtausim->GetName());
    outfile->WriteObject(rtaumax,   rtaumax->GetName());
    outfile->WriteObject(rlambda,   rlambda->GetName());
    outfile->WriteObject(rrelcoeff, rrelcoeff->GetName());
    outfile->WriteObject(repsilon,  repsilon->GetName());
    outfile->WriteObject(rngrid,    rngrid->GetName());
    outfile->WriteObject(rbsize,    rbsize->GetName());
    outfile->WriteObject(rnblock,   rnblock->GetName());
    outfile->WriteObject(rghorder,  rghorder->GetName());
    outfile->WriteObject(rseed,     rseed->GetName());
    outfile->WriteObject(rnthreads, rnthreads->GetName());
    outfile->WriteObject(rbiter,    rbiter->GetName());
    outfile->WriteObject(rnunrel,   rnunrel->GetName());
    outfile->WriteObject(rb_full,   rb_full->GetName());
    outfile->WriteObject(rb_zero,   rb_zero->GetName());
    outfile->WriteObject(rbetasim,  "betasim");
    outfile->Close();
    delete outfile;
    delete hnll;
    delete rtime_s;
    delete rtausim;
    delete rtaumax;
    delete rlambda;
    delete rrelcoeff;
    delete repsilon;
    delete rngrid;
    delete rbsize;
    delete rnblock;
    delete rghorder;
    delete rseed;
    delete rnthreads;
    delete rbiter;
    delete rnunrel;
    delete rb_full;
    delete rb_zero;
    delete rbetasim;
    for(auto h : hbnll) delete h;
  }

  delete psi;
  
  return 0;
}

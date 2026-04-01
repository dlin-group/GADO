#include "GHAModel.hh"
#include <TMath.h>
#include <Math/GSLRndmEngines.h>
#include <H5Cpp.h>
#include <assert.h>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <Rmath.h>
#include <cmath>
#ifdef __CUDA__
#include "CUDAUtils.hh"
#include "gha_kernels.h"
#endif

using namespace std;
using namespace std::chrono;

GHAModel::GHAModel(string ghpath, string ghbase, size_t ghdim,
		   size_t ghorder, bool ghstep,
		   PMatrix<double>* t, PMatrix<double>* d,
		   PMatrix<double>* x, BDMatrix<double>* p){
  for(size_t i=ghdim; i>0; i--)
    gauss_hermite.push_back(GaussHermiteND(ghpath, ghbase, ghorder, i));
  if(ghstep) gh_step = ghorder;
  else gh_step = 0;
  gh_step_max       = 20;
  gh_step_iter_conv = 1;
  // setup clocking
  vector<string> parts({
      "Initialize", "SetSeedsAP", "TimeSortDataAP", "UpdateAP_aq",
      "UpdateAP_w1q", "UpdateAP_ebxw1q", "UpdateAP_beta", "UpdateAP_lambda",
      "UpdateAP_clambda", "UpdateAP_tau",
      "IterateAP", "CheckConvergenceAP",
      "SetSeedsIn", "TimeSortDataIn", "UpdateIn_b", "UpdateIn_mus",
      "UpdateIn_ebxw1", "UpdateIn_w1", "UpdateIn_beta", "UpdateIn_lambda",
      "UpdateIn_clambda", "IterateIn_b", "IterateIn", "CheckConvergenceIn"});
  for(auto const& p : parts) titer_us[p]  = 0.0;
  vector<string> parts_ist({
      "UpdateUniqueAP", "ComputeAP_EV", "ScoreAP_tau",
      "ScoreAP_beta", "ScoreAP_h", "InformationAP", "ScoreInfoAP",
      "UpdateUniqueIn", "ComputeIn_EV", "ComputeIn_EV_GH", "ScoreIn_tau",
      "ScoreIn_beta", "ScoreIn_h", "InformationIn", "ScoreInfoIn",
      "ComputeStheta", "SetG", "ComputeScoreTest", "ComputeVarRatio",
      "ComputePValue"});
  for(auto const& p : parts_ist) titer_ist_us[p] = 0.0;
  ns_fit_ap = 0; ns_fit_in = 0; ns_si_ap = 0; ns_si_in = 0; ns_st = 0;
  auto t0 = high_resolution_clock::now();
  // check dimensions
  assert(t->GetNRows() == d->GetNRows());
  assert(t->GetNCols() == 1 && d->GetNCols() == 1);
  assert(x->GetNRows() == t->GetNRows());
  assert(p->GetNRows() == x->GetNRows() && p->GetNCols() == x->GetNRows());
  // set data variables
  nind = t->GetNRows();
  npar = x->GetNCols();
  b_last        = true;
  b_fullnr      = true;
  centralize_G  = true;
  utransform_G  = false;
  bnorm_ugamma  = false;
  cache_SI      = true;
  verbose       = true;
  T_in          = t;
  delta_in      = d;
  X_in          = x;
  psi_in        = p;
  use_var_ratio = false;
  var_ratio_user= false;
  spa_thresh    = numeric_limits<double>::max();
  vr_init_eval  = 30;
  vr_step_eval  = 10;
  vr_epsilon    = 0.001;
  G_in          = new PMatrix<double>("G_in", T_in->GetNData(), 1);
  // expand psi into the pair structure
  cout << "expanding psi into all pairs structure..." << endl;
  auto t2 = high_resolution_clock::now();
  psi = new BDMatrix<double>("psi_ap");
  PMatrix<double>* block  = new PMatrix<double>();
  PMatrix<double>* sblock = new PMatrix<double>("sblock", 2, 2);
  size_t offset = 0;
  vector<size_t> block_map;
  for(size_t i=0; i<psi_in->GetNBlocks(); i++){
    psi_in->GetBlock(i, *block);
    if(block->GetNRows() <= gauss_hermite.size()){
      size_t n0 = psi->GetNBlocks();
      psi->AppendBlock(block);
      size_t tmp = index_map.size();
      index_map.resize(tmp + block->GetNRows());
      iota(index_map.begin()+tmp, index_map.end(), offset);
      for(size_t l=n0; l<psi->GetNBlocks(); l++) block_map.push_back(i);
    }
    else{
      for(size_t j=0; j<block->GetNRows(); j++)
	for(size_t k=j+1; k<block->GetNRows(); k++){
	  sblock->SetVal(0, 0, block->GetVal(j, j));
	  sblock->SetVal(0, 1, block->GetVal(j, k));
	  sblock->SetVal(1, 0, block->GetVal(k, j));
	  sblock->SetVal(1, 1, block->GetVal(k, k));
	  size_t n0 = psi->GetNBlocks();
	  psi->AppendBlock(sblock);
	  index_map.push_back(offset+j);
	  index_map.push_back(offset+k);
	  for(size_t l=n0; l<psi->GetNBlocks(); l++) block_map.push_back(i);
	}
    }
    offset += block->GetNRows();
  }
  if(block)  delete block;
  if(sblock) delete sblock;
  auto t3 = high_resolution_clock::now();
  cout << "  " << psi->GetNBlocks() << " blocks in "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
       << "expanding input data into all pairs structure..." << endl;
  t2 = high_resolution_clock::now();
  // setup expanded input data
  T     = new PMatrix<double>("T",     psi->GetNRows(), 1);
  delta = new PMatrix<double>("delta", psi->GetNRows(), 1);
  X     = new PMatrix<double>("X",     psi->GetNRows(), X_in->GetNCols());
  G     = new PMatrix<double>("G",     psi->GetNRows(), 1);
  LM.resize(psi->GetNBlocks());
  Asc   = new PMatrix<double>("Asc",   psi_in->GetNBlocks(), 1);
  for(size_t i=0; i<index_map.size(); i++){
    size_t j = index_map[i];
    T->SetVal(i, T_in->GetVal(j));
    delta->SetVal(i, delta_in->GetVal(j));
    for(size_t k=0; k<X->GetNCols(); k++)
      X->SetVal(i, k, X_in->GetVal(j, k));
    G->SetVal(i, G_in->GetVal(j));
  }
  t3 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
       << "sorting blocks..." << endl;
  t2 = high_resolution_clock::now();
  // sort the blocks by size, reorder index map, and permute expanded data
  psi->SortBlocks();
  for(size_t i=0; i<block_map.size(); i++)
    LM[i] = block_map[psi->GetBlockPermutation(i)];
  t3 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
       << "permuting expanded input data..." << endl;
  t2 = high_resolution_clock::now();
  vector<size_t> index_tmp(index_map.begin(), index_map.end());
  fill(index_map.begin(), index_map.end(), 0);
  vector<size_t> index_perm = psi->GetIndexPermutation();
  transform(index_perm.begin(), index_perm.end(), index_map.begin(),
	    [this,&index_tmp](const size_t i){ return index_tmp[i]; });
  T->PermuteRows(index_perm);
  delta->PermuteRows(index_perm);
  X->PermuteRows(index_perm);
  G->PermuteRows(index_perm);
  t3 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
       << "inverting input psi..." << endl;
  // evaluate inverse and sqrt
  t2 = high_resolution_clock::now();
  psiinv_in = new BDMatrix<double>("psiinv_in", *psi_in);
  psiinv_in->Invert();
  t3 = high_resolution_clock::now();
  cout << "  " << scientific << setprecision(4) <<
    duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
       << "inverting pairwise psi..." << endl;
  psiinv = new BDMatrix<double>("psiinv", *psi);
  psiinv->Invert();
  t2 = high_resolution_clock::now();
  cout << "  " << duration_cast<nanoseconds>(t2-t3).count()*1.0e-9
       << " s" << endl << "evaluating sqrt of pairwise psi..." << endl;
  sqrtpsi = psi->Sqrt();
  t3 = high_resolution_clock::now();
  if(verbose)
    cout << "  "
	 << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
	 << "creating sub-matrices with constant block size..." << endl;
  t2 = high_resolution_clock::now();
  // create block matrices with fixed block size
  offset_cbs.resize(gauss_hermite.size(), 0);
  boffset_cbs.resize(gauss_hermite.size(), 0);
  psiinv_cbs.resize(gauss_hermite.size(), NULL);
  sqrtpsi_cbs.resize(gauss_hermite.size(), NULL);
  Ebb_cbs.resize(gauss_hermite.size(), NULL);
  delta_cbs.resize(gauss_hermite.size(), NULL);
  aq_cbs.resize(gauss_hermite.size(), NULL);
  buff0_cbs.resize(gauss_hermite.size(), NULL);
  buff1_cbs.resize(gauss_hermite.size());
  Eebb_cbs.resize(gauss_hermite.size(), NULL);
  Ebbbb_cbs.resize(gauss_hermite.size(), NULL);
  Ebbeb_cbs.resize(gauss_hermite.size());
  size_t bindex = 0;
  offset = 0;
  PMatrix<double>* tmpbl = new PMatrix<double>();
  const size_t N = psiinv_cbs.size();
  for(size_t i=0; i<N; i++){
    const size_t n = N - i;
    offset_cbs[i] = offset;
    boffset_cbs[i] = bindex;
    psiinv_cbs[i]  = new BDMatrix<double>("psiinv_cbs_"+to_string(n));
    sqrtpsi_cbs[i] = new BDMatrix<double>("sqrtpsi_cbs_"+to_string(n));
    while(bindex < psiinv->GetNBlocks()){
      if(psiinv->GetBlockRows(bindex) == n){
	psiinv->GetBlock(bindex, *tmpbl);
	psiinv_cbs[i]->AppendBlock(tmpbl);
	sqrtpsi->GetBlock(bindex, *tmpbl);
	sqrtpsi_cbs[i]->AppendBlock(tmpbl);
	offset += tmpbl->GetNRows();
	bindex ++;
      }
      else break;
    }
    if(sqrtpsi_cbs[i]->GetNBlocks() == 0) continue;
    Ebb_cbs[i] = new BDMatrix<double>("Ebb_cbs_"+to_string(n), *sqrtpsi_cbs[i]);
    Ebb_cbs[i]->Zero();
    delta_cbs[i] = new PMatrix<double>("delta_cbs_"+to_string(n),
				       sqrtpsi_cbs[i]->GetNRows(), 1);
    for(size_t j=0; j<delta_cbs[i]->GetNData(); j++)
      delta_cbs[i]->SetVal(j, delta->GetVal(offset_cbs[i]+j));
    aq_cbs[i] = new PMatrix<double>("aq_cbs_"+to_string(n),
				    sqrtpsi_cbs[i]->GetNRows(), 1);
    size_t p = gauss_hermite[i].GetPowers()[0];
    buff0_cbs[i] = new PMatrix<double>("buff0_cbs_"+to_string(n),
				       sqrtpsi_cbs[i]->GetNRows(), 1);
    buff0_cbs[i]->SetUseCblas(false);
    buff1_cbs[i].resize(p, NULL);
    vector<double> rs2;
    for(size_t j=0; j<p; j++){
      buff1_cbs[i][j] = new PMatrix<double>("buff1_cbs_"+to_string(n),
					    sqrtpsi_cbs[i]->GetNRows(), 1);
      buff1_cbs[i][j]->SetUseCblas(false);
      gauss_hermite[i].GetRootSqrt2Permutation(rs2, j);
      for(size_t k=0; k<buff0_cbs[i]->GetNData(); k++)
	buff0_cbs[i]->SetVal(k, rs2[k%gauss_hermite[i].GetDimension()]);
      sqrtpsi_cbs[i]->RightMult(*buff0_cbs[i], buff1_cbs[i][j]); 
    }
    Eebb_cbs[i] =new BDMatrix<double>("Eebb_cbs_"+to_string(n),*sqrtpsi_cbs[i]);
    Eebb_cbs[i]->Zero();
    Ebbbb_cbs[i] = new BDMatrix<double>("Ebbbb_cbs_"+to_string(n),
					*sqrtpsi_cbs[i]);
    Ebbbb_cbs[i]->Zero();
    for(size_t j=0; j<n; j++){
      Ebbeb_cbs[i].push_back(new BDMatrix<double>("Ebbeb_"+to_string(n)+"_"+
						  to_string(j),
						  *sqrtpsi_cbs[i]));
      Ebbeb_cbs[i].back()->SetUseCblas(false);
    }
    psiinv_cbs[i]->SetUseCblas(false);
    sqrtpsi_cbs[i]->SetUseCblas(false);
    Ebb_cbs[i]->SetUseCblas(false);
    delta_cbs[i]->SetUseCblas(false);
    aq_cbs[i]->SetUseCblas(false);
    Eebb_cbs[i]->SetUseCblas(false);
    Ebbbb_cbs[i]->SetUseCblas(false);
  }
  delete tmpbl;
  t3 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
       << "allocating arrays for computation..." << endl;
  t2 = high_resolution_clock::now();
  // initialize parameters
  converged_ap    = false;
  converged_in    = false;
  niterations     = 0;
  niterations_ap  = 0;
  niterations_in  = 0;
  epsilon         = 1.0e-5;
  absolute_thresh = false;
  beta_ridge      = 0.0;
  beta_ridge_min  = 1e-6;
  beta_ridge_max  = 1e-1;
  beta_weight_min = 1e-12;
  beta_weight_max = 1e12;
  beta_step_max   = 2.0;
  beta_step_l2    = 3.0;
  beta_max        = 50.0;
  betax_max       = 500.0;
  taulimit        = 10.0;
  tau             = 0.0;
  tauprev         = 0.0;
  gamma           = 0.0;
  utau            = 0.0;
  utau_in         = 0.0;
  score_stat      = 0.0;
  score_stat_cent = 0.0;
  ugamma          = 0.0;
  ugamma_cent     = 0.0;
  ugamma_var      = 0.0;
  MAF             = 0.0;
  MAC             = 0.0;
  G_mean          = 0.0;
  vr_mean         = 0.0;
  var_ratio_found = false;
  // initialize input data matrices
  ebx_in        = new PMatrix<double>("ebx_in",     nind,          1);
  ebxw1_in      = new PMatrix<double>("ebxw1_in",   nind,          1);
  w1_in         = new PMatrix<double>("w1_in",      nind,          1);
  lambda_in     = new PMatrix<double>("lambda_in",  nind,          1);
  lambdaprev_in = new PMatrix<double>("lambdaprev_in",nind,        1);
  clambda_in    = new PMatrix<double>("clambda_in", nind,          1);
  b_in          = new PMatrix<double>("b_in",       nind,          1);
  bprev_in      = new PMatrix<double>("bprev_in",   nind,          1);
  ub_in         = new PMatrix<double>("ub_in",      nind,          1);
  mu_in         = new PMatrix<double>("mu_in",      nind,          1);
  sigma_in      = new PMatrix<double>("sigma_in",   nind,          1);
  buffvec_in    = new PMatrix<double>("buffvec_in", nind,          1);
  buffmat_in    = new BDMatrix<double>("buffmat_in", *psi_in);
  buffmat_in->Zero();
  T_in_sort     = new PMatrix<double>("T_in_sort",  nind,           1);
  X_in_sort     = new PMatrix<double>("X_in_sort",  nind,        npar);
  delta_in_sort = new PMatrix<double>("delta_in_sort",nind,         1);
  ebxw1_in_sort = new PMatrix<double>("ebxw1_in_sort",nind,         1);
  Ebbeb_in      = new PMatrix<double>("Ebbeb_in",     nind,         1);
  Ebb_in        = new BDMatrix<double>("Ebb_in", *psi_in);
  Ebb_in->Zero();
  Eebb_in       = new BDMatrix<double>("Eebb_in", *psi_in);
  Eebb_in->Zero();
  Ubeta_in      = new PMatrix<double>("Ubeta_in",   1,             npar);
  // initialize all pairs matrices
  aq            = new PMatrix<double>("aq",         T->GetNRows(), 1);
  w1q           = new PMatrix<double>("w1q",        T->GetNRows(), 1);
  beta          = new PMatrix<double>("beta",       1,             npar);
  betaprev      = new PMatrix<double>("betaprev",   1,             npar);
  beta0         = new PMatrix<double>("beta0",      1,             npar);
  beta1         = new PMatrix<double>("beta1",      1,             npar);
  Ubeta         = new PMatrix<double>("Ubeta",      1,             npar);
  Vbeta         = new PMatrix<double>("Vbeta",      npar,          npar);
  ebx           = new PMatrix<double>("ebx",        T->GetNRows(), 1);
  ebxw1q        = new PMatrix<double>("ebxw1q",     T->GetNRows(), 1);
  lambda        = new PMatrix<double>("lambda",     T->GetNRows(), 1);
  lambdaprev    = new PMatrix<double>("lambdaprev", T->GetNRows(), 1);
  clambda       = new PMatrix<double>("clambda",    T->GetNRows(), 1);
  T_sort        = new PMatrix<double>("T_sort",     T->GetNRows(), 1);
  X_sort        = new PMatrix<double>("X_sort",     T->GetNRows(), npar);
  delta_sort    = new PMatrix<double>("delta_sort", T->GetNRows(), 1);
  ebxw1q_sort   = new PMatrix<double>("ebxw1q_sort",T->GetNRows(), 1);
  X->SetUseCblas(false);
  X_sort->SetUseCblas(false);
  X_in->SetUseCblas(false);
  X_in_sort->SetUseCblas(false);
  Ubeta->SetUseCblas(false);
  Vbeta->SetUseCblas(false);
  T_unq       = NULL;
  h_unq       = NULL;
  Uh_unq      = NULL;
  V_unq       = NULL;
  Vinv_unq    = NULL;
  h_unq_in    = NULL;
  Uh_unq_in   = NULL;
  V_unq_in    = NULL;
  Vinv_unq_in = NULL;
  G_transform = NULL;
  #ifdef __CUDA__
  cumalloc((void**) &w1q_dev, w1q->GetNData()*sizeof(double),
	   "GHAModel", w1q->GetName());
  V_unq_buff  = NULL;
  V_unq_dev   = NULL;
  ebx_dev     = NULL;
  clambda_dev = NULL;
  X_dev       = NULL;
  T_dev       = NULL;
  T_unq_dev   = NULL;
  psiinv_cbs_dev.resize(psiinv_cbs.size(), NULL);
  Ebb_cbs_dev.resize(Ebb_cbs.size(), NULL);
  delta_cbs_dev.resize(delta_cbs.size(), NULL);
  aq_cbs_dev.resize(aq_cbs.size(), NULL);
  tau_cbs_dev.resize(aq_cbs.size(), NULL);
  reduce_cbs_dev.resize(aq_cbs.size(), NULL);
  reduce_cbs.resize(aq_cbs.size());
  buff0_cbs_dev.resize(buff0_cbs.size(), NULL);
  buff1_cbs_dev.resize(buff1_cbs.size());
  buff2_cbs_dev.resize(buff1_cbs.size());
  dtpb_dev.resize(Ebb_cbs.size(), NULL);
  atpb_dev.resize(dtpb_dev.size(), NULL);
  sf0_dev.resize(dtpb_dev.size(), NULL);
  Eebb_cbs_dev.resize(Ebb_cbs.size(), NULL);
  S2_cbs_dev.resize(Ebb_cbs.size(), NULL);
  for(size_t i=0; i<delta_cbs.size(); i++){
    cumalloc((void**) &psiinv_cbs_dev[i],
	     psiinv_cbs[i]->GetNData()*sizeof(double),
	     "GHAModel", psiinv_cbs[i]->GetName());
    cumemcpy((void*) psiinv_cbs_dev[i],
	     (void*) &psiinv_cbs[i]->GetData().front(),
	     psiinv_cbs[i]->GetNData()*sizeof(double),
	     true, "GHAModel", psiinv_cbs[i]->GetName());
    cumalloc((void**) &Ebb_cbs_dev[i],
	     Ebb_cbs[i]->GetNData()*sizeof(double),
	     "GHAModel", Ebb_cbs[i]->GetName());
    cumalloc((void**) &delta_cbs_dev[i],
	     delta_cbs[i]->GetNData()*sizeof(double),
	     "GHAModel", delta_cbs[i]->GetName());
    cumemcpy((void*) delta_cbs_dev[i], (void*) &delta_cbs[i]->GetData().front(),
	     delta_cbs[i]->GetNData()*sizeof(double),
	     true, "GHAModel", delta_cbs[i]->GetName());
    cumalloc((void**) &aq_cbs_dev[i], aq_cbs[i]->GetNData()*sizeof(double),
	     "GHAModel", aq_cbs[i]->GetName());
    cumalloc((void**) &tau_cbs_dev[i], Ebb_cbs[i]->GetNBlocks()*sizeof(double),
	     "GHAModel", "tau_cbs");
    reduce_cbs[i].resize((Ebb_cbs[i]->GetNBlocks()+GHA_THREADS_PER_BLOCK-1) /
			 GHA_THREADS_PER_BLOCK);
    cumalloc((void**) &reduce_cbs_dev[i], reduce_cbs[i].size()*sizeof(double),
	     "GHAModel", "reduce_cbs");
    cumalloc((void**) &buff0_cbs_dev[i],
	     buff0_cbs[i]->GetNData()*sizeof(double),
	     "GHAModel", buff0_cbs[i]->GetName());
    cumalloc((void**) &dtpb_dev[i], sqrtpsi_cbs[i]->GetNBlocks()*sizeof(double),
	     "GHAModel", "dtpb");
    cumalloc((void**) &atpb_dev[i], sqrtpsi_cbs[i]->GetNBlocks()*sizeof(double),
	     "GHAModel", "atpb");
    cumalloc((void**) &sf0_dev[i], sqrtpsi_cbs[i]->GetNBlocks()*sizeof(double),
	     "GHAModel", "sf0");
    cumalloc((void**) &Eebb_cbs_dev[i], Eebb_cbs[i]->GetNData()*sizeof(double),
	     "GHAModel", Eebb_cbs[i]->GetName());
    buff1_cbs_dev[i].resize(buff1_cbs[i].size(), NULL);
    buff2_cbs_dev[i].resize(buff1_cbs[i].size(), NULL);
    for(size_t j=0; j<buff1_cbs[i].size(); j++){
      cumalloc((void**) &buff1_cbs_dev[i][j],
	       buff1_cbs[i][j]->GetNData()*sizeof(double),
	       "GHAModel", buff1_cbs[i][j]->GetName());
      cumemcpy((void*) buff1_cbs_dev[i][j],
	       (void*) &buff1_cbs[i][j]->GetData().front(),
	       buff1_cbs[i][j]->GetNData()*sizeof(double),
	       true, "GHAModel", buff1_cbs[i][j]->GetName());
      cumalloc((void**) &buff2_cbs_dev[i][j],
	       buff1_cbs[i][j]->GetNData()*sizeof(double),
	       "GHAModel", buff1_cbs[i][j]->GetName());
    }
  }
  gG         = NULL;
  gutG       = NULL;
  gStheta    = NULL;
  gIgamma    = NULL;
  gI_G       = NULL;
  gSthetaI_G = NULL;
  #endif
  t3 = high_resolution_clock::now();
  cout << "  "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl;
  auto t1 = high_resolution_clock::now();
  titer_us["InitializeAP"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


GHAModel::~GHAModel(){
  if(ns_st > 0){
  for(auto& pr : titer_ist_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << pr.second << " us  "
	   << pr.second/niterations << " us/iter" << endl;
      pr.second = 0.0;
    }
  }
  if(G_in)          delete G_in;
  if(psiinv_in)     delete psiinv_in;
  if(ebx_in)        delete ebx_in;
  if(ebxw1_in)      delete ebxw1_in;
  if(w1_in)         delete w1_in;
  if(lambda_in)     delete lambda_in;
  if(lambdaprev_in) delete lambdaprev_in;
  if(clambda_in)    delete clambda_in;
  if(b_in)          delete b_in;
  if(bprev_in)      delete bprev_in;
  if(ub_in)         delete ub_in;
  if(mu_in)         delete mu_in;
  if(sigma_in)      delete sigma_in;
  if(buffvec_in)    delete buffvec_in;
  if(buffmat_in)    delete buffmat_in;
  if(Ebb_in)        delete Ebb_in;
  if(Eebb_in)       delete Eebb_in;
  if(Ebbeb_in)      delete Ebbeb_in;
  if(Ubeta_in)      delete Ubeta_in;
  if(T)             delete T;
  if(delta)         delete delta;
  if(G)             delete G;
  if(psi)           delete psi;
  if(psiinv)        delete psiinv;
  if(sqrtpsi)       delete sqrtpsi;
  if(aq)            delete aq;
  if(w1q)           delete w1q;
  if(beta)          delete beta;
  if(beta0)         delete beta0;
  if(beta1)         delete beta1;
  if(betaprev)      delete betaprev;
  if(Ubeta)         delete Ubeta;
  if(Vbeta)         delete Vbeta;
  if(ebx)           delete ebx;
  if(ebxw1q)        delete ebxw1q;
  if(lambda)        delete lambda;
  if(lambdaprev)    delete lambdaprev;
  if(clambda)       delete clambda;
  if(T_sort)        delete T_sort;
  if(X_sort)        delete X_sort;
  if(delta_sort)    delete delta_sort;
  if(ebxw1q_sort)   delete ebxw1q_sort;
  if(T_unq)         delete T_unq;
  if(h_unq)         delete h_unq;
  if(Uh_unq)        delete Uh_unq;
  if(V_unq)         delete V_unq;
  if(Vinv_unq)      delete Vinv_unq;
  if(h_unq_in)      delete h_unq_in;
  if(Uh_unq_in)     delete Uh_unq_in;
  if(V_unq_in)      delete V_unq_in;
  if(Vinv_unq_in)   delete Vinv_unq_in;
  if(Asc)           delete Asc;
  if(G_transform)   delete G_transform;
  for(auto p : Stheta_cache) if(p) delete p;
  for(auto p : Igamma_cache) if(p) delete p;
  for(auto p : psiinv_cbs)   if(p) delete p;
  for(auto p : sqrtpsi_cbs)  if(p) delete p;
  for(auto p : Ebb_cbs)      if(p) delete p;
  for(auto p : delta_cbs)    if(p) delete p;
  for(auto p : aq_cbs)       if(p) delete p;
  for(auto p : buff0_cbs)    if(p) delete p;
  for(auto p : Eebb_cbs)     if(p) delete p;
  for(auto p : Ebbbb_cbs)    if(p) delete p;
  for(auto v : Ebbeb_cbs) for(auto p : v) if(p) delete p;
  for(auto v : buff1_cbs) for(auto p : v) if(p) delete p;
  #ifdef __CUDA__
  if(V_unq_buff)    delete V_unq_buff;
  if(w1q_dev)       cudaFree(w1q_dev);
  if(V_unq_dev)     cudaFree(V_unq_dev);
  if(ebx_dev)       cudaFree(ebx_dev);
  if(clambda_dev)   cudaFree(clambda_dev);
  if(X_dev)         cudaFree(X_dev);
  if(T_dev)         cudaFree(T_dev);
  if(T_unq_dev)     cudaFree(T_unq_dev);
  if(gG)            delete gG;
  if(gutG)          delete gutG;
  if(gStheta)       delete gStheta;
  if(gIgamma)       delete gIgamma;
  if(gI_G)          delete gI_G;
  if(gSthetaI_G)    delete gSthetaI_G;
  for(auto p : psiinv_cbs_dev) if(p) cudaFree(p);
  for(auto p : Ebb_cbs_dev)    if(p) cudaFree(p);
  for(auto p : delta_cbs_dev)  if(p) cudaFree(p);
  for(auto p : aq_cbs_dev)     if(p) cudaFree(p);
  for(auto p : tau_cbs_dev)    if(p) cudaFree(p);
  for(auto p : reduce_cbs_dev) if(p) cudaFree(p);
  for(auto p : buff0_cbs_dev)  if(p) cudaFree(p);
  for(auto p : dtpb_dev)       if(p) cudaFree(p);
  for(auto p : atpb_dev)       if(p) cudaFree(p);
  for(auto p : sf0_dev)        if(p) cudaFree(p);
  for(auto p : Eebb_cbs_dev)   if(p) cudaFree(p);
  for(auto p : S2_cbs_dev)     if(p) cudaFree(p);
  for(auto v : buff1_cbs_dev)  for(auto p : v) if(p) cudaFree(p);
  for(auto v : buff2_cbs_dev)  for(auto p : v) if(p) cudaFree(p);
  #endif
}


GaussHermiteND GHAModel::GetGaussHermite(size_t dim) const{
  if(gauss_hermite.size() == 0) return GaussHermiteND();
  else if(dim >= gauss_hermite.size()) return gauss_hermite[0];
  else return gauss_hermite[dim];
}


void GHAModel::TimeSortData(string label,
			    vector<size_t>&  p, vector<size_t>&  p_t,
			    PMatrix<double>* t, PMatrix<double>* t_sort,
			    PMatrix<double>* x, PMatrix<double>* x_sort,
			    PMatrix<double>* d, PMatrix<double>* d_sort,
			    PMatrix<double>* e, PMatrix<double>* e_sort){
  auto t0 = high_resolution_clock::now();
  auto t2 = t0, t3 = t0;
  string type = "";
  if(label == "TimeSortDataAP")      type = " all pairs";
  else if(label == "TimeSortDataIn") type = " input";
  if(p.size() != t->GetNData()){
    cout << "time sorting" << type << " data..." << endl;
    p.resize(t->GetNData());
    iota(p.begin(), p.end(), 0);
    sort(p.begin(), p.end(),
	 [t](size_t i, size_t j){ return t->GetVal(i) > t->GetVal(j); });
    p_t.resize(t->GetNData());
    for(size_t i=0; i<p.size(); i++) p_t[p[i]] = i;
    t3 = high_resolution_clock::now();
    cout << "  "
	 << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
	 << "permuting rows in" << type << " data..." << endl;
    t2 = high_resolution_clock::now();
    copy(t->GetDataConst().begin(), t->GetDataConst().end(),
	 t_sort->GetData().begin());
    t_sort->Permute(p);
    copy(x->GetDataConst().begin(), x->GetDataConst().end(),
	 x_sort->GetData().begin());
    x_sort->PermuteRows(p);
    copy(d->GetDataConst().begin(), d->GetDataConst().end(),
	 d_sort->GetData().begin());
    d_sort->Permute(p);
  }
  copy(e->GetDataConst().begin(), e->GetDataConst().end(),
       e_sort->GetData().begin());
  e_sort->Permute(p);
  if(t2 != t0){
    t3 = high_resolution_clock::now();
    cout << "  "
	 << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl;
  }
  auto t1 = high_resolution_clock::now();
  titer_us[label]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateBeta(string label,
			  PMatrix<double>* x,      PMatrix<double>* expbx,
			  PMatrix<double>* t_sort, PMatrix<double>* x_sort,
			  PMatrix<double>* d_sort, PMatrix<double>* e_sort){
  auto t0 = high_resolution_clock::now();
  betaprev->SetData(beta->GetData());
  Ubeta->Zero();
  Vbeta->Zero();
  beta_ridge = 0.0;
  size_t j = 0;
  double tmp = 0.0, tmpval = 0.0, denom = 0.0;
  PMatrix<double>* tmprow = new PMatrix<double>("tmprow", *Ubeta);
  PMatrix<double>* tmpmat = new PMatrix<double>("tmpmat", *Vbeta);
  tmprow->SetUseCblas(false); tmpmat->SetUseCblas(false);
  const vector<double>& t  = t_sort->GetDataConst();
  const vector<double>& d  = d_sort->GetDataConst();
  const vector<double>& ew = e_sort->GetDataConst();
  const size_t n = t_sort->GetNData();
  for(size_t i=0; i<n; ){
    while(i == j || (i > 0 ? t[i-1] == t[i] : true)){
      tmp = std::min(std::max((isfinite(ew[i])?ew[i]:0.0),
			      beta_weight_min), beta_weight_max);
      tmpval += tmp;
      tmprow->AddMultRow(0, i, tmp, *x_sort);
      x_sort->OuterRowProduct(i, i, tmp, tmpmat);
      i ++;
      if(i == n) break;
    }
    for(; j<i; j++){
      if(d[j] == 0.0) continue;
      denom = (isfinite(tmpval) && tmpval>0.0) ? tmpval : beta_weight_min;
      Ubeta->AddMultRow(0, j, 1.0, *x_sort);
      Ubeta->AddMultRow(0, 0, -1/denom, *tmprow);
      Vbeta->axpy(1/denom, *tmpmat);
      tmprow->OuterRowProduct(0, 0, -1/denom/denom, Vbeta);
    }
  }
  if(tmpmat) delete tmpmat;
  tmpmat = Vbeta->Inverse();
  while(!tmpmat->IsFinite() && beta_ridge <= beta_ridge_max){
    if(beta_ridge == 0.0) beta_ridge = beta_ridge_min;
    tmpmat->SetData(Vbeta->GetData());
    tmpmat->AddIdentity(beta_ridge);
    for(size_t i=0; i<tmpmat->GetNRows(); i++)
      if(!isfinite(tmpmat->GetVal(i, i)))
	tmpmat->SetVal(i, i, beta_ridge);
    tmpmat->Invert();
    beta_ridge *= 10;
  }
  Ubeta->RightMult(*tmpmat, tmprow);
  tmprow->ApplyLimits(-beta_step_max, beta_step_max);
  tmprow->LimitL2Norm(beta_step_l2);
  (*beta) += (*tmprow);
  beta->ApplyLimits(-beta_max, beta_max);
  for(size_t i=0; i<ebx->GetNData(); i++)
    expbx->SetVal(i, exp(x->InnerRowProduct(i, 0, *beta)));
  expbx->ApplyLimits(exp(-betax_max), exp(betax_max));
  if(tmprow) delete tmprow;
  if(tmpmat) delete tmpmat;
  auto t1 = high_resolution_clock::now();
  titer_us[label] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateLambda(string label,
			    PMatrix<double>* l,      PMatrix<double>* lprev,
			    PMatrix<double>* t_sort, PMatrix<double>* d_sort,
			    PMatrix<double>* e_sort){
  auto t0 = high_resolution_clock::now();
  lprev->SetData(l->GetData());
  l->Zero();
  size_t j = 0;
  double tmp = 0.0;
  const vector<double>& t  = t_sort->GetDataConst();
  const vector<double>& d  = d_sort->GetDataConst();
  const vector<double>& ew = e_sort->GetDataConst();
  const size_t n = t_sort->GetNData();
  for(size_t i=0; i<n; ){
    while(i == j || (i > 0 ? t[i-1] == t[i] : true)){
      tmp += ew[i];
      i ++;
      if(i == n) break;
    }
    for(; j<i; j++) l->SetVal(j, d[j]/tmp);
  }
  auto t1 = high_resolution_clock::now();
  titer_us[label]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateCLambda(string label,            vector<size_t>&  p_t,
			     PMatrix<double>* l,      PMatrix<double>* c,
			     PMatrix<double>* t_sort){
  auto t0 = high_resolution_clock::now();
  c->Zero();
  double tmp = 0.0;
  const vector<double>& t  = t_sort->GetDataConst();
  const size_t n = t_sort->GetNData();
  size_t j = n - 1;
  for(size_t i=n-1; (i>=0 && i<n) || i==(size_t)(-1); ){
    while(i == j || (i<n-1 ? t[i+1]==t[i] : true)){
      tmp += l->GetVal(i);
      i --;
      if(i > n) break;
    }
    for(; i>n?j>=0:j>i; j--){
      c->SetVal(j, tmp);
      if(j == 0) break;
    }
  }
  c->PermuteRows(p_t);
  auto t1 = high_resolution_clock::now();
  titer_us[label]+=duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
}


pair<PMatrix<double>*, PMatrix<double>*>
GHAModel::UpdateUnique(string label,
		       vector<size_t>& imap,    PMatrix<double>* d_sort,
		       PMatrix<double>* t_sort, PMatrix<double>* t_unq,
		       PMatrix<double>* l,      PMatrix<double>* l_unq){
  auto t0 = high_resolution_clock::now();
  if(imap.size() == 0){
    for(size_t i=0; i<t_sort->GetNData(); i++){
      if(d_sort->GetVal(i) == 0.0) continue;
      if((imap.size() > 0) ? (t_sort->GetVal(imap.back()) ==
			      t_sort->GetVal(i)) : false)
	imap.back() = i;
      else
	imap.push_back(i);
    }
    if(t_unq) delete T_unq;
    if(l_unq) delete l_unq;
    t_unq = new PMatrix<double>("T_unq", imap.size(), 1);
    string hname = "h_unq";
    if(label == "UpdateUniqueIn") hname += "_in";
    l_unq = new PMatrix<double>(hname, imap.size(), 1);
  }
  // lambda (l) is already sorted without _sort label
  for(size_t i=0; i<imap.size(); i++){
    t_unq->SetVal(i, t_sort->GetVal(imap[i]));
    l_unq->SetVal(i, l->GetVal(imap[i]));
  }
  auto t1 = high_resolution_clock::now();
  titer_ist_us[label] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
  return make_pair(t_unq, l_unq);
}


void GHAModel::SetSeedsAP(){
  auto t0 = high_resolution_clock::now();
  assert(epsilon > 0.0 && epsilon < 1.0);
  // reset timers
  for(auto& t : titer_us) if(t.first != "Initialize") t.second = 0.0;
  // zero parameters
  converged_ap = false;
  niterations  = 0;
  gh_step_iter.resize(0);
  gh_step_iter.push_back(0);
  gamma       = 0.0;
  tau         = 0.5;
  tauprev     = 0.0;
  beta->Zero();
  ebx->SetAll(1.0);
  var_ratio_found = false;
  // initialize lambda
  TimeSortData("TimeSortDataAP", T_perm, T_perm_t,
	       T, T_sort, X, X_sort, delta, delta_sort, ebxw1q, ebxw1q_sort);
  double dsum = accumulate(delta_sort->GetDataConst().begin(),
			   delta_sort->GetDataConst().end(), 0.0);
  const size_t n = delta_sort->GetNData();
  for(size_t i=0; i<n; i++)
    lambda->SetVal(i, delta_sort->GetVal(i) / dsum);
  UpdateCLambda("UpdateAP_clambda", T_perm_t, lambda, clambda, T_sort);
  UpdateAP_aq();
  tau_iter.resize(0);
  tau_iter.push_back(tau);
  beta_iter.assign(beta->GetNData(), vector<double>(1, 0.0));
  auto t1 = high_resolution_clock::now();
  titer_us["SetSeedsAP"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateAP_aq(){
  auto t0 = high_resolution_clock::now();
  for(size_t i=0; i<aq->GetNData(); i++)
    aq->SetVal(i, clambda->GetVal(i) * ebx->GetVal(i));
  for(size_t i=0; i<aq_cbs.size(); i++){
    PMatrix<double>* aqc = aq_cbs[i];
    if(!aqc) continue;
    for(size_t j=0; j<aqc->GetNData(); j++)
      aqc->SetVal(j, aq->GetVal(offset_cbs[i]+j));
    #ifdef __CUDA__
    cumemcpy((void*) aq_cbs_dev[i], (void*) &aqc->GetData().front(),
	     aqc->GetNData()*sizeof(double), true,"UpdateAP_aq",aqc->GetName());
    #endif
  }
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_aq"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateAP_ebxw1q(){
  auto t0 = high_resolution_clock::now();
  for(size_t i=0; i<ebxw1q->GetNData(); i++)
    ebxw1q->SetVal(i, ebx->GetVal(i) * w1q->GetVal(i));
  if(gamma != 0.0)
    for(size_t i=0; i<ebxw1q->GetNData(); i++)
      ebxw1q->SetVal(i, ebxw1q->GetVal(i) * exp(gamma * G->GetVal(i)));
  TimeSortData("TimeSortDataAP", T_perm, T_perm_t,
	       T, T_sort, X, X_sort, delta, delta_sort, ebxw1q, ebxw1q_sort);
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_ebxw1q"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateAP_w1q(){
  auto t0 = high_resolution_clock::now();
  w1q->Zero();
  #ifdef __CUDA__
  cumemset((void*) w1q_dev, 0,
	   w1q->GetNData()*sizeof(double), "UpdateAP_w1q", "w1q_dev");
  #endif
  const double stau = sqrt(tau);
  for(size_t i=0; i<gauss_hermite.size(); i++){
    if(sqrtpsi_cbs[i]->GetNBlocks() == 0) continue;
    GaussHermiteND gh = gauss_hermite[i];
    BDMatrix<double>* spsi = sqrtpsi_cbs[i];
    const size_t p = gh.GetPowers()[0], n = gh.GetDimension();
    const size_t N = spsi->GetNBlocks(), offset = offset_cbs[i];
    #ifdef __CUDA__
    double alpha = 1.0;
    cumemset((void*) sf0_dev[i], 0, N*sizeof(double), "UpdateAP_w1q", "sf0");
    for(size_t j=0; j<p; j++){
      cudaMemcpy((void*) buff2_cbs_dev[i][j], (void*) buff1_cbs_dev[i][j],
		 buff1_cbs[i][j]->GetNData()*sizeof(double),
		 cudaMemcpyDeviceToDevice);
      cublasDscal(chandle, buff1_cbs[i][j]->GetNData(), &stau,
		  (double*) buff2_cbs_dev[i][j], 1);
      ComponentExponential(buff2_cbs_dev[i][j],
			   buff0_cbs_dev[i], buff0_cbs[i]->GetNData());
      InnerProdStride(delta_cbs_dev[i], buff2_cbs_dev[i][j], dtpb_dev[i], n, N);
      InnerProdStride(aq_cbs_dev[i],    buff0_cbs_dev[i],    atpb_dev[i], n, N);
      DifferenceExponential(dtpb_dev[i], atpb_dev[i], gh.GetWeightProd(j), N);
      cublasDaxpy(chandle, N, &alpha, dtpb_dev[i], 1, sf0_dev[i], 1);
      ProdMod(buff0_cbs_dev[i], dtpb_dev[i], w1q_dev+offset, n, n*N);
    }
    DivideMod(w1q_dev+offset, sf0_dev[i], n, spsi->GetNRows());
    #else
    PMatrix<double>* d = delta_cbs[i], *a = aq_cbs[i], *b0 = buff0_cbs[i], *b1;
    vector<double> dtpb, atpb, sf0(N);
    double dt = 0.0;
    for(size_t j=0; j<p; j++){
      b1 = buff1_cbs[i][j];
      *b1 *= stau;
      auto b0_it = b0->GetData().end(), b1_it = b1->GetData().begin();
      auto w1q_it = w1q->GetData().begin() + offset;
      for(auto b_it=b0->GetData().begin(); b_it!=b0_it; (b_it++, b1_it++))
	*b_it = exp(*b1_it);
      StridedInnerProduct(d, b1, n, dtpb);
      StridedInnerProduct(a, b0, n, atpb);
      b0_it = b0->GetData().begin();
      for(size_t k=0; k<N; k++){
	dt = gh.GetWeightProd(j) * exp(dtpb[k] - atpb[k]);
	sf0[k] += dt;
	for(size_t l=0; l<n; (l++, b0_it++, w1q_it++))
	  *w1q_it += *b0_it * dt;
      }
      *b1 *= 1 / stau;
    }
    auto w1q_it = w1q->GetData().begin() + offset;
    for(size_t j=0; j<spsi->GetNRows(); (j++, w1q_it++)) *w1q_it /= sf0[j/n];
    #endif
  }
  #ifdef __CUDA__
  cumemcpy((void*) &w1q->GetData().front(), (void*) w1q_dev,
	   w1q->GetNData()*sizeof(double),
	   false, "UpdateAP_w1q", w1q->GetName());
  #endif
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_w1q"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;  
}


void GHAModel::UpdateAP_tau(){
  auto t0 = high_resolution_clock::now();
  const double stau = sqrt(tau);
  tauprev = tau;
  tau = 0.0;
  for(size_t i=0; i<gauss_hermite.size(); i++){
    if(sqrtpsi_cbs[i]->GetNBlocks() == 0) continue;
    GaussHermiteND gh = gauss_hermite[i];
    const size_t p = gh.GetPowers()[0], n = gh.GetDimension(), n2 = n * n;
    BDMatrix<double>* ebb = Ebb_cbs[i];
    const size_t N = ebb->GetNBlocks();
    #ifdef __CUDA__
    double alpha = 1.0;
    cumemset((void*) sf0_dev[i], 0, N*sizeof(double), "UpdateAP_tau", "sf0");
    cumemset((void*) Ebb_cbs_dev[i], 0, Ebb_cbs[i]->GetNData()*sizeof(double),
	     "UpdataAP_tau", Ebb_cbs[i]->GetName());
    for(size_t j=0; j<p; j++){
      cudaMemcpy((void*) buff2_cbs_dev[i][j], (void*) buff1_cbs_dev[i][j],
		 buff1_cbs[i][j]->GetNData()*sizeof(double),
		 cudaMemcpyDeviceToDevice);
      cublasDscal(chandle, buff1_cbs[i][j]->GetNData(), &stau,
		  (double*) buff2_cbs_dev[i][j], 1);
      ComponentExponential(buff2_cbs_dev[i][j],
			   buff0_cbs_dev[i], buff0_cbs[i]->GetNData());
      InnerProdStride(delta_cbs_dev[i], buff2_cbs_dev[i][j], dtpb_dev[i], n, N);
      InnerProdStride(aq_cbs_dev[i],    buff0_cbs_dev[i],    atpb_dev[i], n, N);
      DifferenceExponential(dtpb_dev[i], atpb_dev[i], gh.GetWeightProd(j), N);
      cublasDaxpy(chandle, N, &alpha, dtpb_dev[i], 1, sf0_dev[i], 1);
      ProdMod2(buff2_cbs_dev[i][j], dtpb_dev[i], Ebb_cbs_dev[i], n, n2, n2*N);
    }
    DivideMod(Ebb_cbs_dev[i], sf0_dev[i], n2, n2*N);
    double tmp = 0.0;
    cublasDdot(chandle, psiinv_cbs[i]->GetNData(),
    	       psiinv_cbs_dev[i], 1, Ebb_cbs_dev[i], 1, &tmp);
    tau += tmp;
    #else
    ebb->Zero();
    PMatrix<double>* d = delta_cbs[i], *a = aq_cbs[i], *b0 = buff0_cbs[i], *b1;
    vector<double> dtpb, atpb, sf0(ebb->GetNBlocks());
    double dt = 0.0;
    for(size_t j=0; j<p; j++){
      b1 = buff1_cbs[i][j];
      *b1 *= stau;
      auto b0_it = b0->GetData().end(), b1_it = b1->GetData().begin();
      auto ebb_it = ebb->GetData().begin();
      for(auto b_it=b0->GetData().begin(); b_it!=b0_it; (b_it++, b1_it++))
	*b_it = exp(*b1_it);
      StridedInnerProduct(d, b1, n, dtpb);
      StridedInnerProduct(a, b0, n, atpb);
      b1_it = b1->GetData().begin();
      for(size_t k=0; k<N; k++){
	dt = gh.GetWeightProd(j) * exp(dtpb[k] - atpb[k]);
	sf0[k] += dt;
	for(size_t l=0; l<n; l++)
	  for(size_t m=0; m<n; (m++, ebb_it++))
	    *ebb_it += (*(b1_it+l)) * (*(b1_it+m)) * dt;
	b1_it += n;
      }
      *b1 *= 1 / stau;
    }
    auto ebb_it = ebb->GetData().begin();
    double sf = 0.0;
    for(size_t k=0; k<N; (k++, ebb_it+=n2)){
      sf = sf0[k];
      if(n == 1) *ebb_it /= sf;
      else transform(ebb_it, ebb_it+n2, ebb_it,
		     [sf](double el) { return el / sf; });
    }
    tau += psiinv_cbs[i]->TraceProductSymm(*ebb);
    #endif
  }
  tau /= psi->GetNRows();
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_tau"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


// bool GHAModel::IterateAP(size_t niter){
//   bool iterfail = false;
//   bool taufail  = false;
//   bool ignore_taulimit = false;
//   for(size_t i=0; i<niter; i++){
//     auto t0 = high_resolution_clock::now();
//     niterations ++;
//     gh_step_iter.back() ++;
//     UpdateAP_w1q();
//     UpdateAP_ebxw1q();
//     double bnorm = GetMatrixDiff(beta, betaprev);
//     double lnorm = GetMatrixDiff(lambda, lambdaprev);
//     if(std::max(bnorm, lnorm) > 1e-4){
//       UpdateBeta("UpdateAP_beta", X, ebx, T_sort, X_sort, delta_sort,ebxw1q_sort);
//       UpdateAP_ebxw1q();
//       UpdateLambda("Update_AP_lambda", lambda, lambdaprev,
//                    T_sort, delta_sort, ebxw1q_sort);
//       UpdateCLambda("UpdateAP_clambda", T_perm_t, lambda, clambda, T_sort);
//     }
//     UpdateAP_aq();
//     UpdateAP_tau();
//     tau_iter.push_back(tau);
//     for(size_t i=0; i<beta->GetNData(); i++)
//       beta_iter[i].push_back(beta->GetVal(i));
//     auto t1 = high_resolution_clock::now();
//     titer_us["IterateAP"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
//     if(verbose)
//       cout << scientific << "iteration " << niterations << " "<< setprecision(3)
// 	   << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us, "
// 	   << "beta_ridge=" << beta_ridge << ":" <<endl;
//     bool converge = CheckConvergenceAP();
//     if(gh_step == 0){
//       if(converge || i==niter-1){
// 	if(converge) return true;
// 	else         return false;
//       }
//     }
//     else{
//       if(converge || i==niter-1 || (!ignore_taulimit && tau>=taulimit)){
// 	cout << "converged or reached tau/iteration/gh order limit at gh order "
// 	     << gh_step_iter.size()*gh_step 
// 	     <<" after " << gh_step_iter.back() << " iterations" << endl;
// 	if((!ignore_taulimit && tau >= taulimit) || i == niter-1){
// 	  if(iterfail || taufail) return false;
// 	  size_t n = niterations;
// 	  vector<size_t> tmp = gh_step_iter;
// 	  SetSeedsAP();
// 	  niterations = n;
// 	  gh_step_iter = tmp;
// 	  if(i == niter-1) iterfail = true;
// 	  else             taufail  = true;
// 	  i = SIZE_MAX;
// 	}
// 	else{
// 	  if(gh_step_iter.back() < gh_step_iter_conv)         return true;
// 	  else if(gh_step_iter.size()*gh_step >= gh_step_max) return true;
// 	}
// 	if(converge) ignore_taulimit = true;
// 	cout << "setting gh order to " << (gh_step_iter.size()+1)*gh_step
// 	     << "..." << endl;
// 	gh_step_iter.push_back(0);
// 	string ghpath = gauss_hermite[0].GetPath();
// 	string ghbase = gauss_hermite[0].GetBase();
// 	size_t ghdim  = gauss_hermite[0].GetDimension();
// 	gauss_hermite.resize(0);
// 	for(size_t j=ghdim; j>0; j--)
// 	  gauss_hermite.push_back(GaussHermiteND(ghpath, ghbase,
// 						 gh_step_iter.size()*gh_step,j));
// 	for(size_t j=0; j<gauss_hermite.size(); j++){
// 	  for(auto& p : buff1_cbs[j]) if(p) delete p;
// 	  size_t p = gauss_hermite[j].GetPowers()[0];
// 	  buff1_cbs[j].resize(p, NULL);
// 	  vector<double> rs2;
// 	  #ifdef __CUDA__
// 	  buff1_cbs_dev[j].resize(p, NULL);
// 	  buff2_cbs_dev[j].resize(p, NULL);
// 	  #endif
// 	  for(size_t k=0; k<p; k++){
// 	    buff1_cbs[j][k] = new PMatrix<double>("buff1_cbs_"+to_string(j),
// 						  sqrtpsi_cbs[j]->GetNRows(),1);
// 	    buff1_cbs[j][k]->SetUseCblas(false);
// 	    gauss_hermite[j].GetRootSqrt2Permutation(rs2, k);
// 	    for(size_t l=0; l<buff0_cbs[j]->GetNData(); l++)
// 	      buff0_cbs[j]->SetVal(l, rs2[l%gauss_hermite[j].GetDimension()]);
// 	    sqrtpsi_cbs[j]->RightMult(*buff0_cbs[j], buff1_cbs[j][k]);
// 	    #ifdef __CUDA__
// 	    if(buff1_cbs_dev[j][k]) cudaFree(buff1_cbs_dev[j][k]);
// 	    cumalloc((void**) &buff1_cbs_dev[j][k],
// 	       buff1_cbs[j][k]->GetNData()*sizeof(double),
// 	       "GHAModel", buff1_cbs[j][k]->GetName());
// 	    cumemcpy((void*) buff1_cbs_dev[j][k],
// 		     (void*) &buff1_cbs[j][k]->GetData().front(),
// 		     buff1_cbs[j][k]->GetNData()*sizeof(double),
// 		     true, "GHAModel", buff1_cbs[j][k]->GetName());
// 	    if(buff2_cbs_dev[j][k]) cudaFree(buff2_cbs_dev[j][k]);
// 	    cumalloc((void**) &buff2_cbs_dev[j][k],
// 		     buff1_cbs[j][k]->GetNData()*sizeof(double),
// 		     "GHAModel", buff1_cbs[j][k]->GetName());
// 	    #endif
// 	  }
// 	}
//       }
//     }
//   }
//   return false;
// }

bool GHAModel::IterateAP(size_t niter){
  bool iterfail = false;
  bool taufail  = false;
  bool ignore_taulimit = false;
  
  // -----------------------------
  // Block coordinate ascent control
  // Outer loop: update tau
  // Inner loop: converge beta/lambda for fixed tau
  // -----------------------------
  const double eps_bl = epsilon;    // beta/lambda stability threshold (relative; your choice)
  
  const double eps_tau = epsilon;   // tau stability threshold (absolute; tune)
  
  const size_t max_outer = niter;    // interpret niter as max number of tau-updates
  const size_t max_inner = niter;      // max beta/lambda iterations per fixed tau (tune)
  
  // Helper lambda: compute beta/lambda diff (relative)
  auto get_bl = [&](){
    double bnorm = GetMatrixDiff(beta, betaprev);
    double lnorm = GetMatrixDiff(lambda, lambdaprev);
    return std::max(bnorm, lnorm);
  };
  
  RESTART_OUTER:
  
  for(size_t outer=0; outer<max_outer; ++outer){
    auto t0_outer = high_resolution_clock::now();
    
    // ---------------------------------------------------------
    // (0) Ensure tau-dependent weights are consistent at current tau
    // ---------------------------------------------------------
    UpdateAP_w1q();
    UpdateAP_ebxw1q();
    
    // ---------------------------------------------------------
    // (1) INNER LOOP: optimize beta/lambda given fixed tau
    // ---------------------------------------------------------
    
    for(size_t inner=0; inner<max_inner; ++inner){
      auto t0 = high_resolution_clock::now();
      
      niterations ++;
      gh_step_iter.back() ++;
      
      // Update weights that do not require tau change (already fixed tau)
      // Note: w1q and ebxw1q already refreshed at start of outer;
      //       inside inner, ebx changes -> need refresh ebxw1q.
      double bl = get_bl();
      if(bl <= eps_bl){
          cout << "INNER converged: beta/lambda, max(||dBeta||,||dLambda||)=" << bl << "\n";
        break;
      }
      
      // ---- Update beta (given tau) ----
      UpdateBeta("UpdateAP_beta", X, ebx, T_sort, X_sort, delta_sort, ebxw1q_sort);
      
      // ebx depends on beta -> refresh ebxw1q (depends on ebx and w1q)
      UpdateAP_ebxw1q();
      
      // ---- Update lambda/clambda ----
      UpdateLambda("Update_AP_lambda", lambda, lambdaprev,
                   T_sort, delta_sort, ebxw1q_sort);
      UpdateCLambda("UpdateAP_clambda", T_perm_t, lambda, clambda, T_sort);
      
      // ---- Update aq (typically depends on current beta/lambda) ----
      UpdateAP_aq();
      
      auto t1 = high_resolution_clock::now();
      titer_us["IterateAP_inner"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
      
    }
    
    // After inner loop, aq should reflect converged beta/lambda (if UpdateAP_aq uses them)
    // If you want to be extra safe, refresh once here:
    UpdateAP_aq();
    
    // ---------------------------------------------------------
    // (2) OUTER STEP: update tau ONCE given converged beta/lambda
    // ---------------------------------------------------------
    tauprev = tau;
    UpdateAP_tau();
    const double dTau = std::fabs(tau - tauprev);
    
    // Refresh tau-dependent weights AFTER tau update
    // (critical for the next outer iteration's inner convergence)
    UpdateAP_w1q();
    UpdateAP_ebxw1q();
    // 
    // // ---------------------------------------------------------
    // // (3) Record traces
    // // ---------------------------------------------------------
    // tau_iter.push_back(tau);
    // for(size_t j=0; j<beta->GetNData(); j++)
    //   beta_iter[j].push_back(beta->GetVal(j));
    
    auto t1_outer = high_resolution_clock::now();
    titer_us["IterateAP_outer"] += duration_cast<nanoseconds>(t1_outer-t0_outer).count()*1.0e-3;
    
    if(verbose){
      cout << scientific << "OUTER " << outer+1
           << " tau_before=" << tauprev
           << " tau=" << tau
           << " |dTau|=" << dTau
           << " time(us)=" << duration_cast<nanoseconds>(t1_outer-t0_outer).count()*1.0e-3
           << endl;
    }
    
    // ---------------------------------------------------------
    // (4) Convergence logic (tau + beta/lambda)
    // beta/lambda already stable by construction; now check tau
    // ------------------------------------
    
    bool converge = (dTau <= eps_tau);
    
    // -----------------------------
    // GH-step=0 case (no GH refinement logic)
    // -----------------------------
    if(gh_step == 0){
      if(converge || outer == max_outer-1){
        return converge;
      }
      continue;
    }
    
    // ---------------------------------------------------------
    // (5) Keep your existing GH refinement / taulimit / restart logic
    // Trigger points now use OUTER iterations (tau updates) rather than inner iterations
    // ---------------------------------------------------------
    if(converge || outer == max_outer-1 || (!ignore_taulimit && tau >= taulimit)){
      cout << "converged or reached tau/iteration/gh order limit at gh order "
           << gh_step_iter.size()*gh_step
           << " after " << gh_step_iter.back() << " iterations" << endl;
      
      // If we hit taulimit or used up iteration budget, you previously restarted with new seeds
      if((!ignore_taulimit && tau >= taulimit) || outer == max_outer-1){
        if(iterfail || taufail) return false;
        
        size_t n = niterations;
        vector<size_t> tmp = gh_step_iter;
        
        SetSeedsAP();
        
        niterations = n;
        gh_step_iter = tmp;
        
        if(outer == max_outer-1) iterfail = true;
        else                     taufail  = true;
        
        // restart the whole outer loop
        goto RESTART_OUTER;
      } else {
        // if converged: honor your original conditions
        if(gh_step_iter.back() < gh_step_iter_conv)         return true;
        else if(gh_step_iter.size()*gh_step >= gh_step_max) return true;
      }
      
      if(converge) ignore_taulimit = true;
      
      cout << "setting gh order to " << (gh_step_iter.size()+1)*gh_step
           << "..." << endl;
      
      // Increase GH order
      gh_step_iter.push_back(0);
      
      string ghpath = gauss_hermite[0].GetPath();
      string ghbase = gauss_hermite[0].GetBase();
      size_t ghdim  = gauss_hermite[0].GetDimension();
      
      gauss_hermite.resize(0);
      for(size_t j=ghdim; j>0; j--)
        gauss_hermite.push_back(GaussHermiteND(ghpath, ghbase,
                                               gh_step_iter.size()*gh_step, j));
      
      // Rebuild buffers (your original code)
      for(size_t j=0; j<gauss_hermite.size(); j++){
        for(auto& p : buff1_cbs[j]) if(p) delete p;
        
        size_t p = gauss_hermite[j].GetPowers()[0];
        buff1_cbs[j].resize(p, NULL);
        
        vector<double> rs2;
        
#ifdef __CUDA__
        buff1_cbs_dev[j].resize(p, NULL);
        buff2_cbs_dev[j].resize(p, NULL);
#endif
        
        for(size_t k=0; k<p; k++){
          buff1_cbs[j][k] = new PMatrix<double>("buff1_cbs_"+to_string(j),
                                                sqrtpsi_cbs[j]->GetNRows(), 1);
          buff1_cbs[j][k]->SetUseCblas(false);
          
          gauss_hermite[j].GetRootSqrt2Permutation(rs2, k);
          for(size_t l=0; l<buff0_cbs[j]->GetNData(); l++)
            buff0_cbs[j]->SetVal(l, rs2[l%gauss_hermite[j].GetDimension()]);
          
          sqrtpsi_cbs[j]->RightMult(*buff0_cbs[j], buff1_cbs[j][k]);
          
#ifdef __CUDA__
          if(buff1_cbs_dev[j][k]) cudaFree(buff1_cbs_dev[j][k]);
          cumalloc((void**) &buff1_cbs_dev[j][k],
                   buff1_cbs[j][k]->GetNData()*sizeof(double),
                   "GHAModel", buff1_cbs[j][k]->GetName());
          
          cumemcpy((void*) buff1_cbs_dev[j][k],
                   (void*) &buff1_cbs[j][k]->GetData().front(),
                   buff1_cbs[j][k]->GetNData()*sizeof(double),
                   true, "GHAModel", buff1_cbs[j][k]->GetName());
          
          if(buff2_cbs_dev[j][k]) cudaFree(buff2_cbs_dev[j][k]);
          cumalloc((void**) &buff2_cbs_dev[j][k],
                   buff1_cbs[j][k]->GetNData()*sizeof(double),
                   "GHAModel", buff1_cbs[j][k]->GetName());
#endif
        }
      }
      
      // IMPORTANT: after GH order changes, restart outer loop and let beta/lambda re-converge
      // because curvature and approximations changed.
      goto RESTART_OUTER;
    }
  }
  
  return false;
}


double GHAModel::GetParamDiff(double val, double prev) const{
  return abs((val-prev) / ((absolute_thresh || val==prev) ? 1.0 : prev));
}


double GHAModel::GetMatrixDiff(PMatrix<double>* val,
			       PMatrix<double>* prev) const{
  double norm = 0.0;
  if(val->GetNData() != prev->GetNData()) return norm;
  for(size_t i=0; i<val->GetNData(); i++)
    norm = std::max(norm, GetParamDiff(val->GetVal(i), prev->GetVal(i)));
  return norm;
}

bool GHAModel::CheckConvergenceAP_TwoStage(bool freeze_bl, double eps_bl){
  auto t0 = high_resolution_clock::now();
  
  double bnorm = GetMatrixDiff(beta, betaprev);
  double lnorm = GetMatrixDiff(lambda, lambdaprev);
  double tnorm = GetParamDiff(tau, tauprev);
  if(verbose){
    beta->Print(beta->GetNData(), 4, "    ");
    cout << "    tau = " << tau << endl << "    "
         << "||beta|| = "   << setprecision(3) << bnorm << ", "
         << "||lambda|| = " << setprecision(3) << lnorm << ", "
         << "|tau| = "      << setprecision(3) << tnorm << ", "
         << "freeze_bl="    << freeze_bl
         << endl;
  }
  auto t1 = high_resolution_clock::now();
  titer_us["CheckConvergenceAP"] =
    duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
  // Stage 2: only tau criterion
  if(freeze_bl){
    return (tnorm < epsilon);
  }
  
  // Stage 1: joint criterion
  if(tnorm < epsilon && std::max(bnorm, lnorm) < eps_bl) return true;
  return false;
} 

bool GHAModel::CheckConvergenceAP(){
  auto t0 = high_resolution_clock::now();
  double bnorm = GetMatrixDiff(beta, betaprev);
  double lnorm = GetMatrixDiff(lambda, lambdaprev);
  double tnorm = GetParamDiff(tau, tauprev);
  if(verbose){
    beta->Print(beta->GetNData(), 4, "    ");
    cout << "    tau = " << tau << endl << "    "
	 << "||beta|| = "   << setprecision(3)   << bnorm << ", "
	 << "||lambda|| = " << setprecision(3)   << lnorm << ", "
	 << "|tau| = "      << setprecision(3)   << tnorm << endl;
  }
  auto t1 = high_resolution_clock::now();
  titer_us["CheckConvergenceAP"]=duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
  if(tnorm < epsilon && std::max(bnorm, lnorm) < 1e-4) return true;
  return false;
}


void GHAModel::ComputeAP_EV(){
  auto t0 = high_resolution_clock::now();
  const double stau = sqrt(tau);
  w1q->Zero();
  double tmp0 = 0.0, tmp1 = 0.0;
  for(size_t i=0; i<gauss_hermite.size(); i++){
    if(sqrtpsi_cbs[i]->GetNBlocks() == 0) continue;
    GaussHermiteND gh = gauss_hermite[i];
    const size_t p = gh.GetPowers()[0], n = gh.GetDimension();
    BDMatrix<double>* psii=psiinv_cbs[i];
    BDMatrix<double>* Ebbbb=Ebbbb_cbs[i], *Ebb =Ebb_cbs[i], *Eebb=Eebb_cbs[i];
    Ebb->Zero(); Eebb->Zero(); Ebbbb->Zero();
    vector<BDMatrix<double>*> Ebbeb = Ebbeb_cbs[i];
    for(size_t j=0; j<Ebbeb.size(); j++) Ebbeb[j]->Zero();
    PMatrix<double>* d = delta_cbs[i], *a = aq_cbs[i], *b0 = buff0_cbs[i], *b1;
    vector<double> dtpb, atpb, sf0(Ebb->GetNBlocks());
    size_t offset = offset_cbs[i];
    for(size_t j=0; j<p; j++){
      b1 = buff1_cbs[i][j];
      *b1 *= stau;
      for(size_t k=0; k<b0->GetNData(); k++) b0->SetVal(k, exp(b1->GetVal(k)));
      StridedInnerProduct(d, b1, n, dtpb);
      StridedInnerProduct(a, b0, n, atpb);
      size_t q = 0;
      for(size_t k=0; k<dtpb.size(); k++){
	dtpb[k] = gh.GetWeightProd(j) * exp(dtpb[k] - atpb[k]);
	sf0[k] += dtpb[k];
	for(size_t l=0; l<n; l++){
	  w1q->SetVal(offset+q+l, w1q->GetVal(offset+q+l) +
		      b0->GetVal(q+l) * dtpb[k]);
	  for(size_t m=0; m<n; m++){
	    tmp0 = b1->GetVal(q+l) * b1->GetVal(q+m) * dtpb[k];
	    Ebb->SetVal(k, l, m, Ebb->GetVal(k, l, m) + tmp0);
	    Eebb->SetVal(k, l, m, Eebb->GetVal(k, l, m) +
			 exp(b1->GetVal(q+l)+b1->GetVal(q+m)) * dtpb[k]);
	    tmp1 = 0.0;
	    for(size_t v=0; v<n; v++){
	      Ebbeb[v]->SetVal(k, l, m, Ebbeb[v]->GetVal(k, l, m) +
			       tmp0 * b0->GetVal(q+v));
	      for(size_t w=0; w<n; w++)
		tmp1 += b1->GetVal(q+v)*b1->GetVal(q+w)*psii->GetVal(k, v, w);
	    }
	    tmp1 *= tmp0;
	    Ebbbb->SetVal(k, l, m, Ebbbb->GetVal(k, l, m) + tmp1);
	  }
	}
	q += n;
      }
      *b1 *= 1 / stau;
    }
    size_t q = 0;
    for(size_t k=0; k<Ebb->GetNBlocks(); k++){
      for(size_t l=0; l<n; l++){
	w1q->SetVal(offset+q+l, w1q->GetVal(offset+q+l) / sf0[k]);
	for(size_t m=0; m<n; m++){
	  Ebb->SetVal(k, l, m, Ebb->GetVal(k, l, m) / sf0[k]);
	  Eebb->SetVal(k, l, m, Eebb->GetVal(k, l, m) / sf0[k]);
	  Ebbbb->SetVal(k, l, m, Ebbbb->GetVal(k, l, m) / sf0[k]);
	  for(size_t v=0; v<n; v++)
	    Ebbeb[v]->SetVal(k, l, m, Ebbeb[v]->GetVal(k, l, m) / sf0[k]);
	}
      }
      q += n;
    }
    #ifdef __CUDA__
    cumemcpy((void*) Ebb_cbs_dev[i], (void*) &Ebb->GetData().front(),
	     Ebb->GetNData()*sizeof(double),
	     true, "ComputeAP_EV", Ebb->GetName());
    cumemcpy((void*) Eebb_cbs_dev[i], (void*) &Eebb->GetData().front(),
	     Eebb->GetNData()*sizeof(double),
	     true, "ComputeAP_EV", Eebb->GetName());
    #endif
  }
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ComputeAP_EV"]+=duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::ScoreAP_beta(){
  auto t0 = high_resolution_clock::now();
  Ubeta->Zero();
  const size_t N  = Vinv_unq->GetNRows() - 1;
  size_t lblock = 0, cblock = 0;
  double tmp0 = 0.0, tmp1 = 0.0;
  PMatrix<double>* Mbeta = new PMatrix<double>("Mbeta", 1, X->GetNCols());
  for(size_t i=0; i<X->GetNRows(); i++){
    tmp0 = delta->GetVal(i) - clambda->GetVal(i) * ebxw1q->GetVal(i);
    Ubeta->AddMultRow(0, i, tmp0, *X);
    lblock = cblock;
    cblock = psiinv->IndexToBlockIndex(i).first;
    if(lblock != cblock || i == X->GetNRows()-1){
      if(i == X->GetNRows()-1) Mbeta->AddMultRow(0, i, tmp0, *X);
      tmp1 = 0.0;
      for(size_t j=0; j<Mbeta->GetNData(); j++)
	tmp1 += Mbeta->GetVal(j) * Vinv_unq->GetVal(N, j);
      Asc->SetVal(LM[lblock], Asc->GetVal(LM[lblock]) + tmp1);
      Mbeta->Zero();
    }
    Mbeta->AddMultRow(0, i, tmp0, *X);
  }
  delete Mbeta;
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreAP_beta"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;  
}


void GHAModel::ScoreAP_h(){
  auto t0 = high_resolution_clock::now();
  if(Uh_unq) Uh_unq->Zero();
  else Uh_unq = new PMatrix<double>("Uh_unq", h_unq->GetNRows(), 1);
  const size_t L = beta->GetNData();
  const size_t M = T_unq->GetNData();
  const size_t N  = L + M;
  size_t j = 0, k = 0;
  double sum = 0.0, tmp0 = 0.0, tmp1 = 0.0;
  double ti  = 0.0, tj   = 0.0;
  for(size_t i=0; i<M; i++){
    ti = T_unq->GetVal(i);
    tmp0 = 0.0;
    for(j=0; j<T_sort->GetNData(); j++){
      tj = T_sort->GetVal(j);
      if(tj < ti) break;
      else{
	if(tj == ti) tmp0 += delta_sort->GetVal(j);
	sum += ebxw1q_sort->GetVal(j);
	tmp1 = Vinv_unq->GetVal(N, L+i);
	tmp1 *= ((tj==ti ? delta_sort->GetVal(j)/h_unq->GetVal(i) : 0.0) -
		 ebxw1q_sort->GetVal(j));
	k = psiinv->IndexToBlockIndex(T_perm[j]).first;
	Asc->SetVal(LM[k], Asc->GetVal(LM[k]) + tmp1);
      }
    }
    Uh_unq->SetVal(i, tmp0/h_unq->GetVal(i) - sum);
  }
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreAP_h"] += duration_cast<nanoseconds>(t1-t0).count()*1e-3;  
}


void GHAModel::ScoreAP_tau(){
  auto t0 = high_resolution_clock::now();
  utau = 0.0;
  size_t k = 0;
  const double itau2 = 1 / (tau * tau);
  const double tmp0 = 0.5 * Vinv_unq->GetVal(Vinv_unq->GetNRows()-1,
					     Vinv_unq->GetNCols()-1);
  #ifndef __CUDA__
  double tmp1 = 0.0;
  PMatrix<double>* block0 = new PMatrix<double>();
  PMatrix<double>* block1 = new PMatrix<double>();
  #endif
  for(size_t i=0; i<psiinv_cbs.size(); i++){
    if(psiinv_cbs[i]->GetNBlocks() == 0) continue;
    const size_t N = Ebb_cbs[i]->GetNBlocks();
    const size_t n = gauss_hermite[i].GetDimension();
    const double ntau = n / -tau;
    k = boffset_cbs[i];
  #ifdef __CUDA__
    double tmp = 0.0;
    cublasDdot(chandle, psiinv_cbs[i]->GetNData(),
	       psiinv_cbs_dev[i], 1, Ebb_cbs_dev[i], 1, &tmp);
    utau += tmp;
    APBX(tau_cbs_dev[i], tmp0*ntau, tmp0*itau2, N);
    cumemcpy((void*) &buff0_cbs[i]->GetData().front(), (void*) tau_cbs_dev[i],
	     N*sizeof(double), false, "ScoreAP_tau", "tau_cbs");
    const vector<double>& b0 = buff0_cbs[i]->GetDataConst();
    for(size_t j=0; j<N; (j++, k++))
      Asc->SetVal(LM[k], Asc->GetVal(LM[k]) + b0[j]);
  }
  #else
    utau += psiinv_cbs[i]->TraceProductSymm(*Ebb_cbs[i]);
    for(size_t j=0; j<N; (j++, k++)){
      psiinv_cbs[i]->GetBlock(j, *block0);
      Ebb_cbs[i]->GetBlock(j, *block1);
      tmp1 = ntau + block0->TraceProductSymm(*block1)*itau2;
      Asc->SetVal(LM[k], Asc->GetVal(LM[k]) + tmp0 * tmp1);
    }
  }
  delete block0;
  delete block1;
  #endif
  utau = 0.5 * (-(T->GetNData()/tau) + utau*itau2);
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreAP_tau"] += duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::InformationAP(){
  auto t0 = high_resolution_clock::now();
  const size_t L = beta->GetNData();
  const size_t M = T_unq->GetNData();
  const size_t N = L + M;
  const double tau2 = tau  * tau;
  const double tau3 = tau2 * tau;
  const double tau4 = tau2 * tau2;
  if(V_unq){
    V_unq->DeleteRows(V_unq->GetNRows()-1, 1);
    V_unq->DeleteCols(V_unq->GetNCols()-1, 1);
    V_unq->Zero();
  }
  else{
    V_unq    = new PMatrix<double>("V_unq",    N,   N);
    Vinv_unq = new PMatrix<double>("Vinv_unq", N+1, N+1);
  }
  #ifdef __CUDA__
  if(V_unq_dev == NULL){
    cumalloc((void**) &V_unq_dev, V_unq->GetNData()*sizeof(double),
	     "InformationAP", "V_unq");
    cumalloc((void**) &ebx_dev, ebx->GetNData()*sizeof(double),
	     "InformationAP", ebx->GetName());
    cumalloc((void**) &clambda_dev, clambda->GetNData()*sizeof(double),
	     "InformationAP", clambda->GetName());
    cumalloc((void**) &X_dev, X->GetNData()*sizeof(double),
	     "InformationAP", X->GetName());
    cumalloc((void**) &T_dev, T->GetNData()*sizeof(double),
	     "InformationAP", T->GetName());
    cumalloc((void**) &T_unq_dev, T_unq->GetNData()*sizeof(double),
	     "InformationAP", T_unq->GetName());
  }
  cumemset((void*) V_unq_dev, 0, V_unq->GetNData()*sizeof(double),
	   "InformationAP", "V_unq");
  cumemcpy((void*) ebx_dev, (void*) &ebx->GetData().front(),
	   ebx->GetNData()*sizeof(double),
	   true, "InformationAP", ebx->GetName());
  cumemcpy((void*) clambda_dev, (void*) &clambda->GetData().front(),
	   clambda->GetNData()*sizeof(double),
	   true, "InformationAP", clambda->GetName());
  cumemcpy((void*) X_dev, (void*) &X->GetData().front(),
	   X->GetNData()*sizeof(double),
	   true, "InformationAP", X->GetName());
  cumemcpy((void*) T_dev, (void*) &T->GetData().front(),
	   T->GetNData()*sizeof(double),
	   true, "InformationAP", T->GetName());
  cumemcpy((void*) T_unq_dev, (void*) &T_unq->GetData().front(),
	   T_unq->GetNData()*sizeof(double),
	   true, "InformationAP", T_unq->GetName());
  #endif
  PMatrix<double>* c = new PMatrix<double>("c", 1, N);
  PMatrix<double>* d = new PMatrix<double>("d", 1, 1);
  PMatrix<double>* b0 = new PMatrix<double>();
  PMatrix<double>* b1 = new PMatrix<double>();
  PMatrix<double>* X2 = new PMatrix<double>("X2", L, L);
  X2->SetUseCblas(false);
  double tmp0 = 0.0, tmp1 = 0.0;
  for(size_t i=0; i<gauss_hermite.size(); i++){
    if(sqrtpsi_cbs[i]->GetNBlocks() == 0) continue;
    const size_t n = gauss_hermite[i].GetDimension();
    const size_t offset = offset_cbs[i];
    BDMatrix<double>* Ebb = Ebb_cbs[i];
    BDMatrix<double>* Eebb = Eebb_cbs[i];
    BDMatrix<double>* Ebbbb = Ebbbb_cbs[i];
    vector<BDMatrix<double>*>& Ebbeb = Ebbeb_cbs[i];
    BDMatrix<double>* psii = psiinv_cbs[i];
    PMatrix<double>* S2 = new PMatrix<double>("S2", n, N);
    S2->SetUseCblas(false);
    #ifdef __CUDA__
    if(S2_cbs_dev[i] == NULL)
      cumalloc((void**) &S2_cbs_dev[i], S2->GetNData()*sizeof(double),
	       "InformationAP", "S2");
    double alpha = 0.0;
    #endif
    size_t q = 0, r = 0, s = 0;
    for(size_t j=0; j<Eebb->GetNBlocks(); j++){
      // set S2 for this block
      for(size_t k=0; k<n; k++){
	r = offset + q + k;
	tmp0 = ebx->GetVal(r);
	for(size_t l=0; l<L; l++)
	  S2->SetVal(k, l, clambda->GetVal(r) * tmp0 * X->GetVal(r, l));
	for(size_t l=0; l<M; l++)
	  S2->SetVal(k, l+L, (T->GetVal(r) >= T_unq->GetVal(l)) ? tmp0 : 0.0);
      }
      #ifdef __CUDA__
      SetS2(S2_cbs_dev[i], ebx_dev, clambda_dev, X_dev, T_dev, T_unq_dev,
	    n, offset+q, L, M, N);
      #endif
      // add l1 for this block to V_unq and c
      psii->GetBlock(j, *b0);
      Ebb->GetBlock(j, *b1);
      tmp0 = b0->TraceProductSymm(*b1);
      for(size_t k=0; k<n; k++){
	tmp1 = w1q->GetVal(offset+q+k);
	for(size_t l=0; l<n; l++){
	  #ifdef __CUDA__
	  alpha = -Eebb->GetVal(j, k, l) + tmp1*w1q->GetVal(offset+q+l);
	  cublasDger(chandle, N, N, &alpha,
	  	     S2_cbs_dev[i]+k*N, 1, S2_cbs_dev[i]+l*N, 1, V_unq_dev, N);
	  #else
	  S2->OuterRowProduct(k, l, -Eebb->GetVal(j,k,l) +
			      tmp1*w1q->GetVal(offset+q+l), V_unq);
	  #endif
	}
	Ebbeb[k]->GetBlock(j, *b1);
	c->AddMultRow(0, k, b0->TraceProductSymm(*b1) - tmp1*tmp0, *S2);
      }
      // add to d for this block
      Ebbbb->GetBlock(j, *b1);
      d->SetVal(0, d->GetVal(0) + (tmp0*tmp0 - b0->TraceProductSymm(*b1))/(4*tau4));
      // add the contributions from l2 to V_unq
      for(size_t k=0; k<n; k++){
	r = offset + q + k;
	tmp1 = ebx->GetVal(r) * w1q->GetVal(r);
	X->OuterRowProduct(r, r, clambda->GetVal(r) * tmp1, X2);
	for(size_t l=L; l<N; l++){
	  s = l - L;
	  if(T->GetVal(r) < T_unq->GetVal(s)) continue;
	  else if(T->GetVal(r) == T_unq->GetVal(s) && delta->GetVal(r) != 0.0)
	    V_unq->SetVal(l, l, V_unq->GetVal(l, l) +
			  delta->GetVal(r)/h_unq->GetVal(s)/h_unq->GetVal(s));
	  for(size_t v=0; v<L; v++){
	    V_unq->SetVal(v, l, V_unq->GetVal(v, l) + tmp1*X->GetVal(r, v));
	    V_unq->SetVal(l, v, V_unq->GetVal(v, l));
	  }
	}
      }
      // add the last entry of the l2 sum to d
      d->SetVal(0, d->GetVal(0) + tmp0 / tau3);
      q += n;
    }
    delete S2;
  }
  #ifdef __CUDA__
  if(!V_unq_buff) V_unq_buff = new PMatrix<double>("V_unq_buff", *V_unq);
  cumemcpy((void*) &V_unq_buff->GetData().front(), (void*) V_unq_dev,
	   V_unq->GetNData()*sizeof(double),
	   false, "InformationAP", "V_unq");
  *V_unq += *V_unq_buff;
  #endif
  *c *= 1 / (2 * tau2);
  d->SetVal(0, d->GetVal(0) - T->GetNData() / (2 * tau2));
  // add the LxL block of the l2 sum to V_unq
  for(size_t i=0; i<X2->GetNRows(); i++)
    for(size_t j=0; j<X2->GetNCols(); j++)
      V_unq->SetVal(i, j, V_unq->GetVal(i, j) + X2->GetVal(i, j));
  // add the c1 row/col and d to V_unq
  V_unq->InsertRows(V_unq->GetNRows(), 0, 1, *c);
  c->InsertCols(c->GetNCols(), 0, 1, *d);
  c->Transpose();
  V_unq->InsertCols(V_unq->GetNCols(), 0, 1, *c);
  Vinv_unq->SetData(V_unq->GetData());
  Vinv_unq->Invert();
  delete c;
  delete d;
  delete X2;
  if(b0) delete b0;
  if(b1) delete b1;
  auto t1 = high_resolution_clock::now();
  titer_ist_us["InformationAP"]+=duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
}


void GHAModel::ScoreInfoAP(){
  auto t0 = high_resolution_clock::now();
  beta->SetData(beta0->GetData());
  for(size_t i=0; i<ebx->GetNData(); i++)
    ebx->SetVal(i, exp(X->InnerRowProduct(i, 0, *beta)));
  UpdateCLambda("UpdateAP_clambda", T_perm_t, lambda, clambda, T_sort);
  UpdateAP_aq();
  auto unq = UpdateUnique("UpdateUniqueAP", index_unq,
			  delta_sort, T_sort, T_unq, lambda, h_unq);
  T_unq = unq.first;
  h_unq = unq.second;
  cout << T_unq->GetNData() << " unique times" << endl;
  ComputeAP_EV();
  UpdateAP_ebxw1q();
  InformationAP();
  Asc->Zero();
  ScoreAP_tau();
  ScoreAP_beta();
  ScoreAP_h();
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreInfoAP"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::SetSeedsIn(){
  auto t0 = high_resolution_clock::now();
  converged_in = 0;
  niterations  = 0;
  gamma = 0.0;
  b_in->Zero();
  mu_in->Zero();
  var_ratio_found = var_ratio_user;
  TimeSortData("TimeSortDataIn", T_in_perm, T_in_perm_t,
	       T_in, T_in_sort, X_in, X_in_sort, delta_in, delta_in_sort,
	       ebxw1_in, ebxw1_in_sort);
  double dsum = accumulate(delta_in->GetDataConst().begin(),
			   delta_in->GetDataConst().end(), 0.0);
  for(size_t i=0; i<lambda_in->GetNData(); i++){
    lambda_in->SetVal(i, delta_in->GetVal(i) / dsum);
    ebx_in->SetVal(i, exp(X_in->InnerRowProduct(i, 0, *beta)));
  }
  UpdateCLambda("UpdateIn_clambda",
		T_in_perm_t, lambda_in, clambda_in, T_in_sort);
  auto t1 = high_resolution_clock::now();
  titer_us["SetSeedsIn"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateIn_b(){
  auto t0 = high_resolution_clock::now();
  bprev_in->SetData(b_in->GetData());
  for(size_t i=0; i<nind; i++)
    ub_in->SetVal(i, clambda_in->GetVal(i) *
		  ebx_in->GetVal(i) * exp(b_in->GetVal(i)));
  psiinv_in->RightMult(*b_in, buffvec_in);
  *buffvec_in *= -1.0 / tau;
  *buffvec_in += *delta_in;
  *buffvec_in -= *ub_in;
  buffmat_in->Zero();
  *buffmat_in -= *psiinv_in;
  *buffmat_in *= 1.0 / tau;
  for(size_t i=0; i<nind; i++)
    buffmat_in->SetVal(i, i, buffmat_in->GetVal(i, i) - ub_in->GetVal(i));
  buffmat_in->Invert();
  buffmat_in->RightMult(*buffvec_in, b_in);
  *b_in *= -1.0;
  *b_in += *bprev_in;
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateIn_b"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateIn_mus(){
  auto t0 = high_resolution_clock::now();
  buffmat_in->SetData(psiinv_in->GetData());
  *buffmat_in *= 1.0 / tau;
  buffmat_in->RightMult(*b_in, mu_in);
  for(size_t i=0; i<nind; i++){
    sigma_in->SetVal(i, 1.0 / buffmat_in->GetVal(i, i));
    mu_in->SetVal(i, -mu_in->GetVal(i) * sigma_in->GetVal(i));
    sigma_in->SetVal(i, sqrt(sigma_in->GetVal(i)));
  }
  *mu_in += *b_in;
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateIn_mus"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateIn_ebxw1(){
  auto t0 = high_resolution_clock::now();
  for(size_t i=0; i<ebxw1_in->GetNData(); i++)
    ebxw1_in->SetVal(i, ebx_in->GetVal(i) * w1_in->GetVal(i));
  if(gamma != 0.0)
    for(size_t i=0; i<ebxw1_in->GetNData(); i++)
      ebxw1_in->SetVal(i, ebxw1_in->GetVal(i) * exp(gamma * G_in->GetVal(i)));
  TimeSortData("TimeSortDataIn", T_in_perm, T_in_perm_t,
	       T_in, T_in_sort, X_in, X_in_sort, delta_in, delta_in_sort,
	       ebxw1_in, ebxw1_in_sort);
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateIn_ebxw1"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::UpdateIn_w1(){
  auto t0 = high_resolution_clock::now();
  GaussHermiteND gh = gauss_hermite.back();
  w1_in->Zero();
  buffvec_in->Zero();
  double arg = 0.0, m = 0.0;
  for(size_t i=0; i<gh.GetOrder(); i++){
    for(size_t j=0; j<nind; j++){
      arg = sigma_in->GetVal(j) * gh.GetRootSqrt2(i) + mu_in->GetVal(j);
      m  = delta_in->GetVal(j) * arg;
      arg = exp(arg);
      m -= clambda_in->GetVal(j) * ebx_in->GetVal(j) * arg;
      m = exp(m) * gh.GetWeight(i);
      buffvec_in->SetVal(j, buffvec_in->GetVal(j) + m);
      w1_in->SetVal(j, w1_in->GetVal(j) + m*arg);
    }
  }
  for(size_t i=0; i<nind; i++){
    arg = w1_in->GetVal(i);
    w1_in->SetVal(i, arg == 0.0 ? arg : arg / buffvec_in->GetVal(i));
  }
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateIn_w1"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


bool GHAModel::IterateIn_b(size_t niter){
  auto t0 = high_resolution_clock::now();
  if(!b_last) b_in->Zero();
  for(size_t iter=0; iter<niter; iter++){
    UpdateIn_b();
    double bnorm = 0.0;
    for(size_t i=0; i<b_in->GetNData(); i++)
      bnorm += abs(b_in->GetVal(i) - bprev_in->GetVal(i));
    if(bnorm < epsilon){
      if(verbose) cout << "b iterations: " << iter << endl;
      auto t1 = high_resolution_clock::now();
      titer_us["IterateIn_b"] = duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
      return true;
    }
  }
  auto t1 = high_resolution_clock::now();
  titer_us["IterateIn_b"] = duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
  return false;
}


bool GHAModel::IterateIn(size_t niter){
  for(size_t i=0; i<niter; i++){
    auto t0 = high_resolution_clock::now();
    niterations ++;
    if(b_fullnr) IterateIn_b(10);
    else IterateIn_b(1);
    UpdateIn_mus();
    UpdateIn_w1();
    UpdateIn_ebxw1();
    UpdateBeta("UpdateIn_beta", X_in, ebx_in, T_in_sort, X_in_sort,
	       delta_in_sort, ebxw1_in_sort);
    UpdateIn_ebxw1();
    UpdateLambda("UpdateIn_lambda", lambda_in, lambdaprev_in,
		 T_in_sort, delta_in_sort, ebxw1_in_sort);
    UpdateCLambda("UpdateIn_clambda",
		  T_in_perm_t, lambda_in, clambda_in, T_in_sort);
    tau_iter.push_back(tau);
    for(size_t i=0; i<beta->GetNData(); i++)
      beta_iter[i].push_back(beta->GetVal(i));
    auto t1 = high_resolution_clock::now();
    titer_us["Iterate"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    if(verbose)
      cout << scientific << "iteration " << niterations << " "<< setprecision(3)
	   << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us:" <<endl;
    bool converge = CheckConvergenceIn();
    if(converge) return true;
    if(tau >= taulimit) return false;
  }
  return false;
}


bool GHAModel::CheckConvergenceIn(){
  auto t0 = high_resolution_clock::now();
  double bnorm = GetMatrixDiff(beta, betaprev);
  double lnorm = GetMatrixDiff(lambda_in, lambdaprev_in);
  if(verbose){
    beta->Print(beta->GetNData(), 4, "    ");
    cout << "    tau = " << tau << endl << "    "
	 << "||beta|| = "   << setprecision(3)   << bnorm << ", "
	 << "||lambda|| = " << setprecision(3)   << lnorm << endl;
  }
  auto t1 = high_resolution_clock::now();
  titer_us["CheckConvergenceIn"]=duration_cast<nanoseconds>(t1-t0).count()*1e-3;
  if(std::max(bnorm, lnorm) < epsilon) return true;
  return false;
}


void GHAModel::ComputeIn_EV(){
  auto t0 = high_resolution_clock::now();
  // compute the expectation values over a single index
  GaussHermiteND gh = gauss_hermite.back();
  w1_in->Zero();
  Ebb_in->Zero();
  Eebb_in->Zero();
  buffvec_in->Zero();
  buffmat_in->Zero();
  PMatrix<double>* block0 = new PMatrix<double>();
  PMatrix<double>* block1 = new PMatrix<double>();
  PMatrix<double>* block2 = new PMatrix<double>();
  PMatrix<double>* block3 = new PMatrix<double>();
  PMatrix<double>* Sigma2 = new PMatrix<double>("Simga2", 2, 2);
  PMatrix<double>* Sigma3 = new PMatrix<double>("Sigma3", 3, 3);
  PMatrix<double>* sqrtS2 = new PMatrix<double>("sqrtS2", 2, 2);
  PMatrix<double>* sqrtS3 = new PMatrix<double>("sqrtS3", 3, 3);
  PMatrix<double>* mu2    = new PMatrix<double>("mu2",    2, 1);
  PMatrix<double>* mu3    = new PMatrix<double>("mu3",    3, 1);
  PMatrix<double>* z2     = new PMatrix<double>("z2",     2, 1);
  PMatrix<double>* z3     = new PMatrix<double>("z3",     3, 1);
  PMatrix<double>* arg2   = new PMatrix<double>("arg2",   2, 1);
  PMatrix<double>* arg3   = new PMatrix<double>("arg3",   3, 1);
  Sigma2->SetUseCblas(false);
  Sigma3->SetUseCblas(false);
  sqrtS2->SetUseCblas(false);
  sqrtS3->SetUseCblas(false);
  mu2->SetUseCblas(false);
  mu3->SetUseCblas(false);
  z2->SetUseCblas(false);
  z3->SetUseCblas(false);
  arg2->SetUseCblas(false);
  arg3->SetUseCblas(false);
  pair<size_t, size_t> bi;
  double arg = 0.0, earg = 0.0, m = 0.0, tmp = 0.0;
  double  b_in_i = 0.0, b_in_r = 0.0, b_in_s = 0.0;
  size_t q = SIZE_MAX, i0 = 0, r = 0, s = 0;
  for(size_t i=0; i<nind; i++){
    bi = psiinv_in->IndexToBlockIndex(i);
    i0 = bi.second;
    // get blocks of the dimension for this index
    if(bi.first != q){
      q = bi.first;
      psiinv_in->GetBlock(q,  *block0);
      buffmat_in->GetBlock(q, *block1);
      buffmat_in->GetBlock(q, *block2);
      if(block0->GetNRows() != block3->GetNData()){
	delete block3;
	block3 = new PMatrix<double>("block3", block0->GetNRows(), 1);
      }
      if(block0->GetNRows() < 5){
	block0->SetUseCblas(false);
	block1->SetUseCblas(false);
	block2->SetUseCblas(false);
	block3->SetUseCblas(false);
      }
    }
    block1->Zero();
    block2->Zero();
    const size_t n = block0->GetNRows();
    if(n <= gauss_hermite.size()){
      ComputeIn_EV_GH(q);
      i += n-1;
      continue;
    }
    b_in_i = b_in->GetVal(i);
    b_in->SetVal(i, 0.0);
    for(size_t j=0; j<gh.GetOrder(); j++){
      z2->SetVal(0, gh.GetRootSqrt2(j));
      z3->SetVal(0, gh.GetRootSqrt2(j));
      // set values where the gh sum is over one index
      arg = sigma_in->GetVal(i) * gh.GetRootSqrt2(j) + mu_in->GetVal(i);
      earg = exp(arg);
      m  = delta_in->GetVal(i) * arg;
      m -= clambda_in->GetVal(i) * ebx_in->GetVal(i) * earg;
      m  = exp(m) * gh.GetWeight(j);
      buffvec_in->SetVal(i, buffvec_in->GetVal(i) + m);
      buffmat_in->SetVal(i, i, buffmat_in->GetVal(i, i) + m);
      block2->SetVal(i0, i0, block2->GetVal(i0, i0) + m);
      w1_in->SetVal(i, w1_in->GetVal(i) + m*earg);
      Ebb_in->SetVal(i, i, Ebb_in->GetVal(i, i) + m*arg*arg);
      Eebb_in->SetVal(i, i, Eebb_in->GetVal(i, i) + m*earg*earg);
      block1->SetVal(i0, i0, block1->GetVal(i0, i0) + m*arg*arg*earg);
      for(size_t i1=0; i1<n; i1++){
	if(i0 == i1) continue;
	r = psiinv_in->BlockIndexToIndex(q, i1);
	b_in_r = b_in->GetVal(r);
	b_in->SetVal(r, 0.0);	
	// compute sigma and mu for the 2d case
	Sigma2->SetVal(0, 0, block0->GetVal(i0, i0));
	Sigma2->SetVal(0, 1, block0->GetVal(i0, i1));
	Sigma2->SetVal(1, 0, block0->GetVal(i1, i0));
	Sigma2->SetVal(1, 1, block0->GetVal(i1, i1));
	Sigma2->Invert();
	b_in->GetBlock(psiinv_in->GetBlockOffset(q), 0, block3);
	arg2->SetVal(0, block0->InnerProduct(i0, 0, *block3));
	arg2->SetVal(1, block0->InnerProduct(i1, 0, *block3));
	Sigma2->RightMult(*arg2, mu2);
	*mu2 *= -1.0;
	*Sigma2 *= tau;
	Sigma2->Sqrt(sqrtS2);
	for(size_t k=0; k<gh.GetOrder(); k++){
	  z2->SetVal(1, gh.GetRootSqrt2(k));
	  z3->SetVal(1, gh.GetRootSqrt2(k));
	  tmp = gh.GetWeight(j) * gh.GetWeight(k);
	  // set values where the gh sum is over 2 indices
	  sqrtS2->RightMult(*z2, arg2);
	  *arg2 += *mu2;
	  m  = delta_in->GetVal(i) * arg2->GetVal(0);
	  m += delta_in->GetVal(r) * arg2->GetVal(1); 
	  m -= clambda_in->GetVal(i)*ebx_in->GetVal(i) * exp(arg2->GetVal(0));
	  m -= clambda_in->GetVal(r)*ebx_in->GetVal(r) * exp(arg2->GetVal(1));
	  m  = exp(m) * tmp;
	  buffmat_in->SetVal(q, i0, i1, buffmat_in->GetVal(q, i0, i1) + m);
	  block2->SetVal(i0, i1, block2->GetVal(i0, i1) + m);
	  block2->SetVal(i1, i0, block2->GetVal(i1, i0) + m);
	  block2->SetVal(i1, i1, block2->GetVal(i1, i1) + m);
	  Ebb_in->SetVal(q, i0, i1, Ebb_in->GetVal(q, i0, i1) +
			 m*arg2->GetVal(0)*arg2->GetVal(1));
	  Eebb_in->SetVal(q, i0, i1, Eebb_in->GetVal(q, i0, i1) +
			  m*exp(arg2->GetVal(0)+arg2->GetVal(1)));
	  block1->SetVal(i0, i1, block1->GetVal(i0, i1) +
			 m*exp(arg2->GetVal(0))*arg2->GetVal(0)*arg2->GetVal(1));
	  block1->SetVal(i1, i0, block1->GetVal(i1, i0) +
	  		 m*exp(arg2->GetVal(0))*arg2->GetVal(1)*arg2->GetVal(0));
	  block1->SetVal(i1, i1, block1->GetVal(i1, i1) +
	  		 m*exp(arg2->GetVal(0))*arg2->GetVal(1)*arg2->GetVal(1));
	  for(size_t i2=0; i2<n; i2++){
	    if(i0 == i2 || i1 == i2) continue;
	    s = psiinv_in->BlockIndexToIndex(q, i2);
	    b_in_s = b_in->GetVal(s);
	    b_in->SetVal(s, 0.0);
	    // compute sigma and mu for the 3d case
	    Sigma3->SetVal(0, 0, block0->GetVal(i0, i0));
	    Sigma3->SetVal(0, 1, block0->GetVal(i0, i1));
	    Sigma3->SetVal(0, 2, block0->GetVal(i0, i2));
	    Sigma3->SetVal(1, 1, block0->GetVal(i1, i1));
	    Sigma3->SetVal(1, 2, block0->GetVal(i1, i2));
	    Sigma3->SetVal(2, 2, block0->GetVal(i2, i2));
	    Sigma3->MakeSymmetric(true);
	    Sigma3->SetUseCblas(true);
	    Sigma3->Invert();
	    Sigma3->SetUseCblas(false);
	    b_in->GetBlock(psiinv_in->GetBlockOffset(q), 0, block3);
	    arg3->SetVal(0, block0->InnerProduct(i0, 0, *block3));
	    arg3->SetVal(1, block0->InnerProduct(i1, 0, *block3));
	    arg3->SetVal(2, block0->InnerProduct(i2, 0, *block3));   
	    Sigma3->RightMult(*arg3, mu3);
	    *mu3 *= -1.0;
	    *Sigma3 *= tau;
	    Sigma3->SetUseCblas(true);
	    Sigma3->Sqrt(sqrtS3);
	    Sigma3->SetUseCblas(false);
	    for(size_t l=0; l<gh.GetOrder(); l++){
	      z3->SetVal(2, gh.GetRootSqrt2(l));
	      // set values where the gh sum is over 3 indices
	      sqrtS3->RightMult(*z3, arg3);
	      *arg3 += *mu3;
	      m  = delta_in->GetVal(i) * arg3->GetVal(0);
	      m += delta_in->GetVal(r) * arg3->GetVal(1);
	      m += delta_in->GetVal(s) * arg3->GetVal(2);
	      m -= clambda_in->GetVal(i)*ebx_in->GetVal(i)*exp(arg3->GetVal(0));
	      m -= clambda_in->GetVal(r)*ebx_in->GetVal(r)*exp(arg3->GetVal(1));
	      m -= clambda_in->GetVal(s)*ebx_in->GetVal(s)*exp(arg3->GetVal(2));
	      m = exp(m) * tmp * gh.GetWeight(l);
	      block2->SetVal(i1, i2, block2->GetVal(i1, i2) + m);
	      block1->SetVal(i1, i2, block1->GetVal(i1, i2) +
			     m*exp(arg3->GetVal(0)) *
			     arg3->GetVal(1)*arg3->GetVal(2));
	    }
	    b_in->SetVal(s, b_in_s);
	  }
	}
	b_in->SetVal(r, b_in_r);
      }
    }
    b_in->SetVal(i, b_in_i);
    // set Ebbeb for this i
    for(size_t j=0; j<n; j++)
      for(size_t k=0; k<n; k++){
	arg = block1->GetVal(j, k);
	block1->SetVal(j, k, arg == 0.0 ? arg : arg/block2->GetVal(j, k));
      }
    Ebbeb_in->SetVal(i, block0->TraceProductSymm(*block1));
  }
  // divide by denominator for w1
  for(size_t i=0; i<nind; i++){
    const size_t q = psiinv_in->IndexToBlockIndex(i).first;
    const size_t n = psiinv_in->GetBlockRows(q);
    if(n <= gauss_hermite.size()) continue;
    arg = w1_in->GetVal(i);
    w1_in->SetVal(i, arg == 0.0 ? arg : arg / buffvec_in->GetVal(i));
  }
  // divide by denominator for Ebb and Eebb or skip for GH blocks
  for(q=0; q<psiinv_in->GetNBlocks(); q++){
    const size_t n = psiinv_in->GetBlockRows(q);
    if(n <= gauss_hermite.size()) continue;
    for(size_t i=0; i<n; i++)
      for(size_t j=0; j<n; j++){
	arg = Ebb_in->GetVal(q, i, j);
	Ebb_in->SetVal(q,i,j, arg==0.0 ? arg : arg/buffmat_in->GetVal(q,i,j));
	arg = Eebb_in->GetVal(q, i, j);
	Eebb_in->SetVal(q,i,j,arg==0.0 ? arg : arg/buffmat_in->GetVal(q,i,j));
      }
  }
  delete block0;
  delete block1;
  delete block2;
  delete block3;
  delete Sigma2;
  delete Sigma3;
  delete sqrtS2;
  delete sqrtS3;
  delete mu2;
  delete mu3;
  delete z2;
  delete z3;
  delete arg2;
  delete arg3;
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ComputeIn_EV"] = duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::ComputeIn_EV_GH(size_t q){
  auto t0 = high_resolution_clock::now();
  const double stau = sqrt(tau);
  const size_t n = psiinv_in->GetBlockRows(q);
  const size_t offset = psiinv_in->GetBlockOffset(q);
  assert(n <= gauss_hermite.size());
  const size_t g = gauss_hermite.size() - n;
  GaussHermiteND gh = gauss_hermite[g];
  const size_t p = gh.GetPowers()[0];
  PMatrix<double>* spsi = new PMatrix<double>("spsi",
					      psi_in->GetBlock(q)->GetNRows(),
					      psi_in->GetBlock(q)->GetNCols());
  PMatrix<double>* psitmp = psi_in->GetBlock(q);
  psitmp->Sqrt(spsi);
  delete psitmp;
  PMatrix<double>* psii = psiinv_in->GetBlock(q);
  for(size_t i=0; i<n; i++){
    w1_in->SetVal(offset+i, 0.0);
    Ebbeb_in->SetVal(offset+i, 0.0);
  }
  Ebb_in->ZeroBlock(q);
  Eebb_in->ZeroBlock(q);
  PMatrix<double>* b0 = new PMatrix<double>("b0", n, 1);
  PMatrix<double>* b1 = new PMatrix<double>("b1", n, 1);
  vector<PMatrix<double>*> b2(n, NULL);
  for(size_t i=0; i<n; i++)
    b2[i] = new PMatrix<double>("b2"+to_string(i), n, n);
  size_t i = 0;
  vector<double> rs2;
  double tmp0 = 0.0, dtpb = 0.0, atpb = 0.0, sf0 = 0.0;
  for(size_t j=0; j<p; j++){
    gh.GetRootSqrt2Permutation(rs2, j);
    for(size_t k=0; k<n; k++) b0->SetVal(k, rs2[k]);
    spsi->RightMult(*b0, b1);
    *b1 *= stau;
    for(size_t k=0; k<n; k++) b0->SetVal(k, exp(b1->GetVal(k)));
    dtpb = 0.0;
    atpb = 0.0;
    for(size_t k=0; k<n; k++){
      i = offset + k;
      dtpb += delta_in->GetVal(i) * b1->GetVal(k);
      atpb += clambda_in->GetVal(i) * ebx_in->GetVal(i) * b0->GetVal(k);
    }
    dtpb = gh.GetWeightProd(j) * exp(dtpb - atpb);
    sf0 += dtpb;
    for(size_t l=0; l<n; l++){
      i = offset + l;
      w1_in->SetVal(i, w1_in->GetVal(i) + b0->GetVal(l) * dtpb);
      for(size_t m=0; m<n; m++){
	tmp0 = b1->GetVal(l) * b1->GetVal(m) * dtpb;
	Ebb_in->SetVal(q, l, m, Ebb_in->GetVal(q, l, m) + tmp0);
	Eebb_in->SetVal(q, l, m, Eebb_in->GetVal(q, l, m) +
			exp(b1->GetVal(l) + b1->GetVal(m)) * dtpb);
	for(size_t v=0; v<n; v++)
	  b2[v]->SetVal(l, m, b2[v]->GetVal(l, m) + tmp0*b0->GetVal(v));
      }
    }
  }
  for(size_t l=0; l<n; l++){
    i = offset + l;
    w1_in->SetVal(i, w1_in->GetVal(i) / sf0);
    for(size_t m=0; m<n; m++){
      Ebb_in->SetVal(q, l, m, Ebb_in->GetVal(q, l, m) / sf0);
      Eebb_in->SetVal(q, l, m, Eebb_in->GetVal(q, l, m) / sf0);
      for(size_t v=0; v<n; v++)
	b2[v]->SetVal(l, m, b2[v]->GetVal(l, m) / sf0);
    }
  }
  for(size_t l=0; l<n; l++)
    Ebbeb_in->SetVal(offset+l, psii->TraceProductSymm(*(b2[l])));
  delete spsi;
  delete psii;
  delete b0;
  delete b1;
  for(auto& p : b2) delete p;
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ComputeIn_EV_GH"]+=duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::ScoreIn_beta(){
  auto t0 = high_resolution_clock::now();
  Ubeta_in->Zero();
  for(size_t i=0; i<X_in->GetNRows(); i++)
    Ubeta_in->AddMultRow(0, i, delta_in->GetVal(i) -
			 clambda_in->GetVal(i) * ebxw1_in->GetVal(i), *X);
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreIn_beta"]+=duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::ScoreIn_h(){
  auto t0 = high_resolution_clock::now();
  if(Uh_unq_in) Uh_unq->Zero();
  else Uh_unq_in = new PMatrix<double>("Uh_unq_in", h_unq_in->GetNRows(), 1);
  size_t j = 0;
  double sum = 0.0, tmp = 0.0;
  double ti  = 0.0, tj  = 0.0;
  for(size_t i=0; i<h_unq_in->GetNData(); i++){
    ti = T_unq->GetVal(i);
    tmp = 0.0;
    for(; j<T_in_sort->GetNData(); j++){
      tj = T_in_sort->GetVal(j);
      if(tj < ti) break;
      else{
	if(tj == ti) tmp += delta_in_sort->GetVal(j);
	sum += ebxw1_in_sort->GetVal(j);
      }
    }
    Uh_unq_in->SetVal(i, tmp/h_unq_in->GetVal(i) - sum);
  }
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreIn_h"] += duration_cast<nanoseconds>(t1-t0).count()*1e-3;  
}


void GHAModel::ScoreIn_tau(){
  auto t0 = high_resolution_clock::now();
  utau_in = psiinv_in->TraceProductSymm(*Ebb_in);
  utau_in = 0.5 * (-(T_in->GetNData()/tau) + utau_in/tau/tau);
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreIn_tau"] = duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::InformationIn(){
  auto t0 = high_resolution_clock::now();
  const size_t L = beta->GetNData();
  const size_t M = T_unq->GetNData();
  const size_t N = T_unq->GetNData() + beta->GetNData();
  const double tau2 = tau * tau;
  if(V_unq_in){
    V_unq_in->DeleteRows(V_unq_in->GetNRows()-1, 1);
    V_unq_in->DeleteCols(V_unq_in->GetNCols()-1, 1);
    V_unq_in->Zero();
  }
  else{
    V_unq_in    = new PMatrix<double>("V_unq_in",    N, N);
    Vinv_unq_in = new PMatrix<double>("Vinv_unq_in", N, N);
  }
  PMatrix<double>* c  = new PMatrix<double>("c", 1, N);
  PMatrix<double>* d  = new PMatrix<double>("d", 1, 1);
  PMatrix<double>* b0 = new PMatrix<double>();
  PMatrix<double>* b1 = new PMatrix<double>();
  PMatrix<double>* X2 = new PMatrix<double>("X2", L, L);
  PMatrix<double>* S2 = new PMatrix<double>();
  c->SetUseCblas(false);
  d->SetUseCblas(false);
  b0->SetUseCblas(false);
  b1->SetUseCblas(false);
  X2->SetUseCblas(false);
  S2->SetUseCblas(false);
  double tmp0 = 0.0, tmp1 = 0.0;
  size_t r = 0, s = 0;
  for(size_t j=0; j<Eebb_in->GetNBlocks(); j++){
    const size_t n = Eebb_in->GetBlockRows(j);
    const size_t offset = Eebb_in->GetBlockOffset(j);
    if(S2->GetNRows() != n){
      delete S2;
      S2 = new PMatrix<double>("S2", n, N);
      S2->SetUseCblas(false);
    }
    else S2->Zero();
    // set S2 for this block
    for(size_t k=0; k<n; k++){
      r = offset + k;
      tmp0 = ebx_in->GetVal(r);
      for(size_t l=0; l<L; l++)
	S2->SetVal(k, l, clambda_in->GetVal(r) * tmp0 * X_in->GetVal(r, l));
      for(size_t l=0; l<M; l++)
	S2->SetVal(k, l+L, (T_in->GetVal(r)>=T_unq->GetVal(l)) ? tmp0 : 0.0);
    }
    // add l1 for this block to V_unq and c
    psiinv_in->GetBlock(j, *b0);
    Ebb_in->GetBlock(j, *b1);
    tmp0 = b0->TraceProductSymm(*b1);
    for(size_t k=0; k<n; k++){
      tmp1 = w1_in->GetVal(offset+k);
      for(size_t l=0; l<n; l++)
	S2->OuterRowProduct(k, l, -Eebb_in->GetVal(j,k,l) +
			    tmp1*w1_in->GetVal(offset+l), V_unq_in);
      c->AddMultRow(0, k, Ebbeb_in->GetVal(offset+k) - tmp1*tmp0, *S2);
    }
    // add the contributions from l2 to V_unq
    for(size_t k=0; k<n; k++){
      r = offset + k;
      tmp1 = ebx_in->GetVal(r) * w1_in->GetVal(r);
      X_in->OuterRowProduct(r, r, clambda_in->GetVal(r) * tmp1, X2);
      for(size_t l=L; l<N; l++){
	s = l - L;
	if(T_in->GetVal(r) < T_unq->GetVal(s)) continue;
	else if(T_in->GetVal(r) == T_unq->GetVal(s) &&
		delta_in->GetVal(r) != 0.0)
	  V_unq_in->SetVal(l, l, V_unq_in->GetVal(l, l) + delta_in->GetVal(r) /
			   h_unq_in->GetVal(s)/h_unq_in->GetVal(s));
	for(size_t v=0; v<L; v++){
	  V_unq_in->SetVal(v, l, V_unq_in->GetVal(v, l) +
			   tmp1*X_in->GetVal(r, v));
	  V_unq_in->SetVal(l, v, V_unq_in->GetVal(v, l));
	}
      }
    }
  }
  *c *= 1 / (2 * tau2);
  // set d to be that from the all pairs step
  d->SetVal(0, V_unq->GetVal(V_unq->GetNRows()-1, V_unq->GetNCols()-1));
  // add the LxL block of the l2 sum to V_unq
  for(size_t i=0; i<X2->GetNRows(); i++)
    for(size_t j=0; j<X2->GetNCols(); j++)
      V_unq_in->SetVal(i, j, V_unq_in->GetVal(i, j) + X2->GetVal(i, j));
  // get the inverse for the score test, excluding the last row and col
  Vinv_unq_in->SetData(V_unq_in->GetData());
  Vinv_unq_in->Invert();
  // add the c1 row/col and d to V_unq
  V_unq_in->InsertRows(V_unq_in->GetNRows(), 0, 1, *c);
  c->InsertCols(c->GetNCols(), 0, 1, *d);
  c->Transpose();
  V_unq_in->InsertCols(V_unq_in->GetNCols(), 0, 1, *c);
  delete c;
  delete d;
  delete X2;
  delete S2;
  if(b0) delete b0;
  if(b1) delete b1;
  auto t1 = high_resolution_clock::now();
  titer_ist_us["InformationIn"]+=duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::ScoreInfoIn(){
  auto t0 = high_resolution_clock::now();
  beta->SetData(beta1->GetData());
  for(size_t i=0; i<ebx_in->GetNData(); i++)
    ebx_in->SetVal(i, exp(X_in->InnerRowProduct(i, 0, *beta)));
  UpdateCLambda("UpdateAP_clambda",
		T_in_perm_t, lambda_in, clambda_in, T_in_sort);
  auto unq = UpdateUnique("UpdateUniqueIn", index_unq_in,
			  delta_in_sort, T_in_sort, T_unq, lambda_in, h_unq_in);
  T_unq    = unq.first;
  h_unq_in = unq.second;
  UpdateIn_mus();
  ComputeIn_EV();
  UpdateIn_ebxw1();
  ScoreIn_tau();
  ScoreIn_beta();
  ScoreIn_h();
  InformationIn();
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreInfoIn"] = duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::Solve(size_t maxiterap, size_t maxiterin){
  auto t0 = high_resolution_clock::now();
  SetSeedsAP();
  if(maxiterap == 0) converged_ap = IterateAP(SIZE_MAX);
  else converged_ap = IterateAP(maxiterap);
  auto t1 = high_resolution_clock::now();
  cout << endl << "Step 1 solution after " << niterations << " iterations, "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s:" << endl;
  beta->Print(beta->GetNData(), 4, "    ");
  cout << "tau = " << tau << endl << endl;
  for(auto& pr : titer_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << pr.second << " us  "
	   << pr.second/niterations << " us/iter" << endl;
      pr.second = 0.0;
    }
  beta0->SetData(beta->GetData());
  ns_fit_ap = duration_cast<nanoseconds>(t1-t0).count();
  niterations_ap = niterations;
  cout << "Fixing tau, iterating for step 2..." << endl;
  t0 = high_resolution_clock::now();
  SetSeedsIn();
  if(maxiterin == 0) converged_in = IterateIn(SIZE_MAX);
  else converged_in = IterateIn(maxiterin);
  t1 = high_resolution_clock::now();
  cout << endl << "Step 2 solution after " << niterations << " iterations, "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s:" << endl;
  beta->Print(beta->GetNData(), 4, "    ");
  cout << "tau = " << tau << endl << endl;
  for(auto& pr : titer_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << pr.second << " us  "
	   << pr.second/niterations << " us/iter" << endl;
      pr.second = 0.0;
    }
  beta1->SetData(beta->GetData());
  ns_fit_in = duration_cast<nanoseconds>(t1-t0).count();
  niterations_in = niterations;
}


void GHAModel::SetG(PMatrix<double>* g){
  auto t0 = high_resolution_clock::now();
  assert(g != NULL);
  if(G_in != g){
    delete G_in;
    G_in = g;
  }
  if(!G_in){
    G->Zero();
    return;
  }
  assert(G_in->GetNData() == T_in->GetNData());
  MAC = accumulate(G_in->GetDataConst().begin(),G_in->GetDataConst().end(),0.);
  MAF = MAC / 2.0 / G_in->GetNData();
  if(MAF > 0.5){
    for(size_t i=0; i<G_in->GetNData(); i++)
      G_in->SetVal(i, 2.0 - G_in->GetVal(i));
    MAC = 2 * G_in->GetNData() - MAC;
    MAF = 1 - MAF;
  }
  if(utransform_G){
    if(!G_transform){
      G_transform = new PMatrix<double>("G_transform", nind, nind);
      PMatrix<double>* X_in_aug = new PMatrix<double>("X_in_aug",
						      X->GetNRows(),
						      X->GetNCols()+1);
      X_in_aug->SetBlock(0, 1, *X_in);
      for(size_t i=0; i<X->GetNRows(); i++) X_in_aug->SetVal(i, 0, 1.0);
      const size_t n = X_in_aug->GetNCols();
      PMatrix<double>* XaugtUXaug = new PMatrix<double>("XaugtUXaug", n, n);
      for(size_t k=0; k<nind; k++){
	double ubk = clambda_in->GetVal(k) *
	  ebx_in->GetVal(k) * exp(b_in->GetVal(k));
	for(size_t i=0; i<n; i++)
	  for(size_t j=i; j<n; j++)
	    XaugtUXaug->SetVal(i, j, XaugtUXaug->GetVal(i, j) +
			       X_in_aug->GetVal(k, i) * ubk *
			       X_in_aug->GetVal(k, j));
      }
      XaugtUXaug->MakeSymmetric(true);
      XaugtUXaug->Invert();
      double tmp = 0.0;
      for(size_t i=0; i<nind; i++)
	for(size_t j=0; j<nind; j++){
	  tmp = 0.0;
	  for(size_t q=0; q<n; q++)
	    for(size_t l=0; l<n; l++)
	      tmp += X_in_aug->GetVal(i, q) * XaugtUXaug->GetVal(q, l)
		* X_in_aug->GetVal(j, l);
	  G_transform->SetVal(i, j, tmp);
	}
      delete X_in_aug;
      delete XaugtUXaug;
    }
    buffvec_in->SetData(G_in->GetData());
    double tmp = 0.0;
    for(size_t i=0; i<nind; i++){
      tmp = 0.0;
      for(size_t j=0; j<nind; j++)
	tmp += G_transform->GetVal(i, j) * buffvec_in->GetVal(j);
      G_in->SetVal(i, buffvec_in->GetVal(i) - tmp);
    }
  }
  G_mean = G_in->Sum() / G_in->GetNData();
  if(centralize_G) for(size_t i=0; i<G_in->GetNData(); i++)
		     G_in->SetVal(i, G_in->GetVal(i) - G_mean);
  for(size_t i=0; i<index_map.size(); i++)
    G->SetVal(i, G_in->GetVal(index_map[i]));
  auto t1 = high_resolution_clock::now();
  titer_ist_us["SetG"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::ComputeScoreInfo(){
  auto t0 = high_resolution_clock::now();
  cout << "Computing AP score and information..." << endl;
  ScoreInfoAP();
  auto t1 = high_resolution_clock::now();
  cout << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;
  cout << "U_tau: " << utau << endl;
  Ubeta->Print();
  Uh_unq->Print();
  ns_si_ap = duration_cast<nanoseconds>(t1-t0).count();
  Vinv_unq->Print();
  cout << "  ";
  for(size_t i=0; i<6; i++)
    cout << Vinv_unq->GetVal(Vinv_unq->GetNRows()-1, i) << " ";
  cout << "..." << endl
       << Vinv_unq->GetVal(Vinv_unq->GetNRows()-1,
			   Vinv_unq->GetNCols()-1) << endl;
  for(auto& pr : titer_ist_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << pr.second << " us" << endl;
      pr.second = 0.0;
    }
  t0 = high_resolution_clock::now();
  cout << "Computing input data score and information..." << endl;
  ScoreInfoIn();
  t1 = high_resolution_clock::now();
  cout << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s" << endl;
  cout << "U_tau: " << utau_in << endl;
  Ubeta_in->Print();
  Uh_unq_in->Print();
  ns_si_in = duration_cast<nanoseconds>(t1-t0).count();
  Vinv_unq_in->Print();
  cout << "  ";
  for(size_t i=0; i<6; i++)
    cout << Vinv_unq_in->GetVal(Vinv_unq_in->GetNRows()-1, i) << " ";
  cout << "..." << endl
       << Vinv_unq_in->GetVal(Vinv_unq_in->GetNRows()-1,
			      Vinv_unq_in->GetNCols()-1) << endl;
  for(auto& pr : titer_ist_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << pr.second << " us" << endl;
      pr.second = 0.0;
    }
  niterations = 0;
}


void GHAModel::ComputeScoreTest(bool print){
  auto t0 = high_resolution_clock::now();
  if(print) cout << "Computing score test, MAC="
		 << MAC << ", MAF=" << MAF << "  ...  ";
  niterations ++;
  ugamma      = 0.0;
  ugamma_cent = 0.0;
  ugamma_var  = 0.0;
  const size_t L = beta->GetNData();
  const size_t M = T_unq->GetNData();
  const size_t N = T_unq->GetNData() + beta->GetNData();
  buffvec_in->Zero();
  // compute ugamma
  for(size_t i=0; i<delta_in->GetNData(); i++)
    buffvec_in->SetVal(i, delta_in->GetVal(i) -
			clambda_in->GetVal(i)*ebxw1_in->GetVal(i));
  vector<double> utG, utcG;
  StridedInnerProduct(buffvec_in, G_in, psiinv_in->GetBlockRows(), utG,
		      0.0, 0.0, bnorm_ugamma);
  double ug_mean = accumulate(buffvec_in->GetData().begin(),
			      buffvec_in->GetData().end(), 0.0);
  ug_mean /= buffvec_in->GetNData();
  for(size_t i=0; i<buffvec_in->GetNData(); i++)
    buffvec_in->SetVal(i, buffvec_in->GetVal(i) - ug_mean);
  StridedInnerProduct(buffvec_in, G_in, psiinv_in->GetBlockRows(), utcG,
		      0.0, (centralize_G ? G_mean : 0.0), bnorm_ugamma);
  // get the ugamma's from the strided products
  ugamma      = accumulate(utG.begin(),  utG.end(),  0.0);
  ugamma_cent = accumulate(utcG.begin(), utcG.end(), 0.0);
  // decide whether to perform the full calculation
  bool full_calc = true;
  if(use_var_ratio && var_ratio_found){
    full_calc = false;
    ugamma_var = ComputeVarRatio();
    if(ugamma_var == 0.0 && (ugamma != 0.0 || ugamma_cent != 0.0)){
      full_calc = true;
      ugamma_var = 0.0;
    }
  }
  // perform the full variance calculation if necessary
  if(full_calc){
    bool d = false;
    size_t k = 0, r = 0, v = 0;
    double tmp0 = 0.0, tmp1 = 0.0, tmp2 = 0.0;
    PMatrix<double>* Stheta = NULL;
    PMatrix<double>* Stmp   = NULL;
    PMatrix<double>* I_G    = NULL;
    PMatrix<double>* Igamma = NULL;
    PMatrix<double>* Gi     = NULL;
    PMatrix<double>* block0 = NULL;
    PMatrix<double>* block1 = NULL;
    bool use_cache = false;
    if(Stheta_cache.size() == psiinv_in->GetNBlocks() &&
       Igamma_cache.size() == psiinv_in->GetNBlocks()){
      delete Stheta;
      delete Igamma;
      use_cache = true;
    }
#ifndef __CUDA__
    if(!use_cache){
      Stheta = new PMatrix<double>("Stheta",   N, 1);
      Igamma = new PMatrix<double>();
    }
    Stmp   = new PMatrix<double>("Stmp",     N, 1);
    I_G    = new PMatrix<double>("I_G",    N+1, 1);
    Gi     = new PMatrix<double>();
    block0 = new PMatrix<double>();
    block1 = new PMatrix<double>();
#else
    if(use_cache){
      for(size_t i=0; i<G_in->GetNData(); i++) gG->SetVal(i, G_in->GetVal(i));
      for(size_t i=0; i<utG.size(); i++) gutG->SetVal(i, utG[i]);
      gI_G->Zero();
      gSthetaI_G->Zero();
      gG->CopyToDevice();
      gutG->CopyToDevice();
      gI_G->CopyToDevice();
      gSthetaI_G->CopyToDevice();
    }
    else{
      Stheta  = new PMatrix<double>("Stheta",   N, 1);
      Stmp    = new PMatrix<double>("Stmp",     N, 1);
      I_G     = new PMatrix<double>("I_G",    N+1, 1);
      Igamma  = new PMatrix<double>();
      Gi      = new PMatrix<double>();
      block0  = new PMatrix<double>();
      block1  = new PMatrix<double>();
      gG      = new GMatrix<double>("gG",G_in->GetNRows(),   1, false, chandle);
      gutG    = new GMatrix<double>("gutG",   utG.size(),    1, false, chandle);
      gStheta = new GMatrix<double>("gStheta",utG.size(),  N+1, false, chandle);
      gIgamma = new GMatrix<double>("gIgamma",       N+1, nind, false, chandle);
      gI_G    = new GMatrix<double>("gI_G",          N+1,    1, false, chandle);
      gSthetaI_G=new GMatrix<double>("gSthetaI_G",utG.size(),1, false, chandle);
      gG->SetNoCopyFromDevice(true);
      gStheta->SetNoCopyFromDevice(true);
      gIgamma->SetNoCopyFromDevice(true);
      gI_G->SetNoCopyFromDevice(true);
      gSthetaI_G->SetNoCopyFromDevice(true);
    }
    if(!use_cache){
#endif
      // compute I_G
      size_t offset = 0;
      for(size_t i=0; i<psiinv_in->GetNBlocks(); i++){
	const size_t n = psiinv_in->GetBlockRows(i);
	if(Gi->GetNRows() != n){
	  delete Gi;
	  Gi = new PMatrix<double>("Gi", n, 1);
	}
	for(size_t s=0; s<n; s++)
	  Gi->SetVal(s, G_in->GetVal(v+s));
	if(use_cache) Igamma = Igamma_cache[i];
	else{
	  if(Igamma->GetNCols() != n){
	    delete Igamma;
	    Igamma = new PMatrix<double>("Igamma", N+1, n);
	  }
	  else Igamma->Zero();
	  psiinv_in->GetBlock(i, *block0);
	  Ebb_in->GetBlock(i, *block1);
	  tmp1 = block0->TraceProductSymm(*block1);
	  for(size_t s=0; s<n; (s++, v++)){
	    tmp0 = clambda_in->GetVal(v) * ebx_in->GetVal(v);
	    for(size_t l=0; l<L; l++)
	      Igamma->SetVal(l, s, w1_in->GetVal(v) * X_in->GetVal(v, l));
	    for(size_t l=0; l<M; l++)
	      if(T_in->GetVal(v) >= T_unq->GetVal(l))
		Igamma->SetVal(l+L, s, w1_in->GetVal(v));
	    Igamma->SetVal(N, s, tmp0 / (2*tau*tau) *
			   (Ebbeb_in->GetVal(v) - w1_in->GetVal(v) * tmp1));
	    r = v - s;
	    for(size_t t=0; t<n; (t++, r++)){
	      tmp2 = -Eebb_in->GetVal(i, s,t)+w1_in->GetVal(v)*w1_in->GetVal(r);
	      for(size_t l=0; l<L; l++)
		Igamma->SetVal(l, s, Igamma->GetVal(l, s) + X_in->GetVal(r, l) *
			       tmp2*clambda_in->GetVal(r)*ebx_in->GetVal(r));
	      for(size_t l=0; l<M; l++)
		if(T_in->GetVal(r) >= T_unq->GetVal(l))
		  Igamma->SetVal(l+L, s, Igamma->GetVal(l+L, s) +
				 tmp2*clambda_in->GetVal(v)*ebx_in->GetVal(r));
	    }
	    for(size_t l=0; l<L; l++)
	      Igamma->SetVal(l, s, Igamma->GetVal(l, s) * tmp0);
	    for(size_t l=L; l<N; l++)
	      Igamma->SetVal(l, s, Igamma->GetVal(l, s) * ebx_in->GetVal(v));
	  }
	  if(!var_ratio_found && cache_SI)
	    Igamma_cache.push_back(new PMatrix<double>("Igamma_"+to_string(i),
						       *Igamma));
#ifdef __CUDA__
	  for(size_t l=0; l<Igamma->GetNRows(); l++)
	    for(size_t j=0; j<Igamma->GetNCols(); j++)
	      gIgamma->SetVal(l, offset+j, Igamma->GetVal(l, j));
#endif
	}
	// for small n, bypass cblas, compute the contribution to I_G directly
	if(n <= 3){
	  const vector<double>& igamma = Igamma->GetDataConst();
	  vector<double>& ig = I_G->GetData();
	  const double g0 = Gi->GetVal(0);
	  if(n == 1)
	    for(size_t l=0; l<Igamma->GetNData(); l++)
	      ig[l] += igamma[l] * g0;
	  else{
	    size_t q = 0;
	    const double g1 = Gi->GetVal(1);
	    if(n == 2)
	      for(size_t l=0; l<Igamma->GetNData(); (l+=2, q++))
		ig[q] += igamma[l] * g0 + igamma[l+1] * g1;
	    else if(n == 3){
	      const double g2 = Gi->GetVal(2);
	      for(size_t l=0; l<Igamma->GetNData(); (l+=3, q++))
		ig[q] += igamma[l] * g0 + igamma[l+1] * g1 + igamma[l+2] * g2;
	    }
	  }
	}
	else Igamma->RightMult(*Gi, I_G, false);
	if(use_cache){
	  Igamma = NULL;
	  v += n;
	}
	offset += n;
      }
      // compute S_theta
      for(size_t i=0; i<psiinv_in->GetNBlocks(); i++){
	if(use_cache) Stheta = Stheta_cache[i];
	else{
	  V_unq_in->GetBlock(0, N, Stmp);
	  *Stmp *= -Asc->GetVal(i);
	  for(size_t j=0; j<psiinv_in->GetBlockRows(i); (j++, k++)){
	    d = delta_in->GetVal(k) != 0.0;
	    tmp0 = ebx_in->GetVal(k);
	    for(size_t l=0; l<L; l++){
	      if(d) Stmp->SetVal(l, Stmp->GetVal(l) + X_in->GetVal(k, l));
	      Stmp->SetVal(l, Stmp->GetVal(l) - w1_in->GetVal(k) *
			   clambda_in->GetVal(k) * tmp0 * X_in->GetVal(k, l));
	    }
	    r = L;
	    for(size_t l=0; l<M; (l++, r++)){
	      if(d) if(T_in->GetVal(k) == T_unq->GetVal(l))
		      Stmp->SetVal(r, Stmp->GetVal(r) +
				   delta_in->GetVal(k) / h_unq_in->GetVal(l));
	      if(T_in->GetVal(k) >= T_unq->GetVal(l))
		Stmp->SetVal(r, Stmp->GetVal(r) - w1_in->GetVal(k) * tmp0);
	    }
	  }
	  Vinv_unq_in->RightMult(*Stmp, Stheta);
	  Stheta->InsertRows(N, i, 1, *Asc);
	  if(!var_ratio_found && cache_SI)
	    Stheta_cache.push_back(new PMatrix<double>("Stheta_"+to_string(i),
						       *Stheta));
#ifdef __CUDA__
	  for(size_t l=0; l<Stheta->GetNData(); l++)
	    gStheta->SetVal(l, i, Stheta->GetVal(l));
#endif	
	}
	tmp2 = utG[i] - Stheta->InnerColProduct(0, 0, *I_G);
	ugamma_var += tmp2 * tmp2 / (bnorm_ugamma ?
				     sqrt(psiinv_in->GetBlockRows(i)) : 1.0);
	if(use_cache) Stheta = NULL;
	else Stheta->DeleteRows(N, 1);
      }
#ifdef __CUDA__
    }
    else{
      gIgamma->RightMult(*gG, gI_G);
      gStheta->RightMult(*gI_G, gSthetaI_G);
      gutG->axpy(-1.0, *gSthetaI_G);
      for(size_t i=0; i<gutG->GetNData(); i++)
	ugamma_var += gutG->GetVal(i) * gutG->GetVal(i) /
	  (bnorm_ugamma ? sqrt(psiinv_in->GetBlockRows(i)) : 1.0);
    }
#endif
    if(Stheta) delete Stheta;
    if(Stmp)   delete Stmp;
    if(I_G)    delete I_G;
    if(Igamma) delete Igamma;
    if(Gi)     delete Gi;
    if(block0) delete block0;
    if(block1) delete block1;
    if(use_var_ratio) ComputeVarRatio();
  }
  // score stat and p-value for non-centralized case
  if(ugamma == 0.0 && ugamma_var == 0.0) score_stat = 0.0;
  else score_stat = ugamma  / sqrt(ugamma_var);
  if((abs(score_stat) < spa_thresh)){
    pval = TMath::Prob(score_stat*score_stat, 1);
  }else{
    pval = ComputeWeightedP(ugamma, ugamma_var);
  }
    
  // score stat and p-value for centralized case
//   if(ugamma_cent == 0.0 && ugamma_var == 0.0) score_stat_cent = 0.0;
//   else score_stat_cent = ugamma_cent / sqrt(ugamma_var);
//   // if((full_calc && (use_var_ratio ||
// 		    (abs(score_stat_cent) <= spa_thresh))) ||
//      (!full_calc && (abs(score_stat_cent) <= spa_thresh)))
//     pval_cent = TMath::Prob(score_stat_cent*score_stat_cent, 1);
//   else
//     pval_cent = ComputePValue(ugamma_cent, ugamma_var);
  // print values and timing
  auto t1 = high_resolution_clock::now();
  size_t ns = duration_cast<nanoseconds>(t1-t0).count();
  ns_st += ns;
  if(print){
    cout << ns * 1.0e-9 << " s" << endl
	 << "p-value:         " << pval            << endl
	 // << "p-value cent:    " << pval_cent       << endl
	 << "score test:      " << score_stat      << endl
	 // << "score test cent: " << score_stat_cent << endl
	 << "Ugamma:          " << ugamma          << endl
	 // << "Ugamma cent:     " << ugamma_cent     << endl
	 << "Ugamma_var:      " << ugamma_var      << endl;
  }
  titer_ist_us["ComputeScoreTest"] += ns * 1e-3;
}


double GHAModel::ComputeVarRatio(){
  auto t0 = high_resolution_clock::now();
  // set u_b if this is the first variance ratio being calculated
  if(var_ratio.size() == 0 && (!var_ratio_found || var_ratio_user)){
    if(ub_in == NULL ? true : (ub_in->GetNData() != nind)){
      if(ub_in) delete ub_in;
      ub_in = new PMatrix<double>("ub_in", nind, 1);
    }
    for(size_t i=0; i<nind; i++)
      ub_in->SetVal(i, clambda_in->GetVal(i) * ebxw1_in->GetVal(i));
    if(var_ratio_user) var_ratio.push_back(vr_mean);
  }
  // calculate GuG
  for(size_t i=0; i<nind; i++)
    buffvec_in->SetVal(i, ub_in->GetVal(i) * G_in->GetVal(i));
  double GuG = G_in->InnerColProduct(0, 0, *buffvec_in);
  // check to see if we've found a sufficient rhat
  if(!var_ratio_found){
    if(std::isfinite(GuG) && GuG > 1e-8 &&
       std::isfinite(ugamma_var) && ugamma_var > 0){
      
      double r = ugamma_var / GuG;
      
      if(std::isfinite(r) && r > 0)
        var_ratio.push_back(r);
      const size_t n = var_ratio.size();
      if(n <= vr_init_eval ?
           (n == vr_init_eval) : ((n - vr_init_eval) % vr_step_eval) == 0){
        
        // ---- compute median ----
        std::vector<double> vr_copy = var_ratio;
        std::sort(vr_copy.begin(), vr_copy.end());
        
        double med;
        size_t m = vr_copy.size();
        if(m % 2 == 0)
          med = 0.5 * (vr_copy[m/2 - 1] + vr_copy[m/2]);
        else
          med = vr_copy[m/2];
        
        cout << "checking for acceptable variance ratio after "
             << n << " evaluations... median = " << med << endl;
        
        // ---- convergence check: median stabilized ----
        if(std::abs(med - vr_prev) / std::max(1e-12, std::abs(vr_prev)) < 0.005){
          var_ratio_found = true;
          vr_mean = std::min(med, 10.0);  
          cout << "found acceptable variance ratio, rhat = "
               << vr_mean << endl;
          
          for(auto s : Stheta_cache) delete s;
          for(auto I : Igamma_cache) delete I;
          Stheta_cache.clear();
          Igamma_cache.clear();
        }
        
        vr_prev = med;
      }
      
    }
  }
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ComputeVarRatio"] = duration_cast<nanoseconds>(t1-t0).count()*1.-3;
  return vr_mean * GuG;
}

double GHAModel::ComputeWeightedP(double ug, double var)
{
  double z = ug / std::sqrt(var);
  double absz = std::fabs(z);
  
  double p_normal = TMath::Prob(z*z, 1);
  double p_spa    = ComputePValue(ug, var);
  
  double c0 = 2.0;
  double delta = 0.5;
  
  double w = (absz - (c0 - delta)) / (2.0 * delta);
  if(w < 0.0) w = 0.0;
  if(w > 1.0) w = 1.0;
  
  return (1.0 - w) * p_normal + w * p_spa;
}

double GHAModel::ComputePValue(const double& ug, const double& var){
  auto t0 = high_resolution_clock::now();
  
  auto one_side_fallback = [&](double /*s*/) {
    // symmetric split of chi-square(1) two-sided p
    double p_two = TMath::Prob((ug*ug)/var, 1);
    return 0.5 * p_two;
  };
  
  double p_two = 0.0;
  
  for(int i=-1; i<=1; i+=2){
    double s  = ug * i;
    
    double k0 = 0.0, k1 = 0.0, k1p = 0.0, k2 = 0.0;
    double c1 = 0.0, c1p = std::numeric_limits<double>::max();
    double alpha = 0.0, alphap = std::numeric_limits<double>::max();
    size_t nrit = 0;
    
    // init
    ComputeCGFs(c1, k0, k1, k2);
    
    while(std::abs(c1 - c1p) >= 1e-4){
      if(nrit++ >= 1000 || !std::isfinite(c1) || std::isnan(c1) || !std::isfinite(k2)){
        // consistent two-sided fallback
        return TMath::Prob((ug*ug)/var, 1);
      }
      
      alpha = (k1 - s) / k2;
      c1p = c1;
      c1 -= alpha;
      k1p = k1;
      
      ComputeCGFs(c1, k0, k1, k2);
      
      if(std::signbit(k1 - s) != std::signbit(k1p - s)){
        if(std::abs(alpha) >= alphap){
          alphap /= 2.0;
          c1 = c1p + (std::signbit(k1 - k1p) ? -1.0 : 1.0) * alphap;
          ComputeCGFs(c1, k0, k1, k2);
        }else{
          alphap = std::abs(alpha);
        }
      }
    }
    
    double p_side = 0.0;
    
    ComputeCGFs(c1, k0, k1, k2);
    
    // fallback if CGF derivatives invalid
    if(!std::isfinite(k1) || !std::isfinite(k2) || k2 <= 0.0){
      p_side = one_side_fallback(s);
    }else{
      double temp = c1*s - k0;
      
      // IMPORTANT: protect sqrt
      if(!std::isfinite(temp) || temp <= 0.0){
        p_side = one_side_fallback(s);
      }else{
        double c2 = (std::signbit(c1) ? -1.0 : 1.0) * std::sqrt(2.0 * temp);
        double c3 = c1 * std::sqrt(k2);
        
        if(!std::isfinite(c2) || !std::isfinite(c3) || std::abs(c2) < 1e-10 || (c3/c2) <= 0.0){
          p_side = one_side_fallback(s);
        }else{
          double Z = c2 + std::log(c3 / c2) / c2;
          
          if(!std::isfinite(Z)){
            p_side = one_side_fallback(s);
          }else{
            // one-sided tail chosen by sign of s (NOT by sign of Z)
            // s>0 => upper tail; s<0 => lower tail
            bool lower_tail = (s < 0.0);
            p_side = R::pnorm(Z, 0.0, 1.0, lower_tail, false);
            
            // numeric guard
            if(!std::isfinite(p_side)) p_side = one_side_fallback(s);
          }
        }
      }
    }
    
    p_two += p_side;
  }
  
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ComputePValue"]=duration_cast<nanoseconds>(t1-t0).count()*1.-3;
  
  // final guard (two-sided)
  if(!std::isfinite(p_two) || p_two < 0.0 || p_two > 1.0) return TMath::Prob((ug*ug)/var, 1);
  return p_two;
}


void GHAModel::ComputeCGFs(const double& c1,
			   double& k0, double& k1, double& k2){
  k0 = 0.0;
  k1 = 0.0;
  k2 = 0.0;
  const size_t n = clambda_in->GetNData();
  double u = 0.0, c1G = 0.0, ec1G = 0.0, G = 0.0;
  for(size_t i=0; i<n; i++){
    u    = ub_in->GetVal(i);
    G    = G_in->GetVal(i);
    c1G  = c1 * G;
    ec1G = exp(c1G);
    k0 += u * (ec1G - c1G - 1);
    k1 += u * G * (ec1G - 1);
    k2 += u * G * G * ec1G;
  }
}


void GHAModel::StridedInnerProduct(PMatrix<double>* v0, PMatrix<double>* v1,
				   size_t stride, vector<double>& out,
				   double off0, double off1, bool snorm){
  const size_t n = v0->GetNData();
  assert(n == v1->GetNData());
  assert(n % stride == 0);
  const size_t m = n / stride;
  out.resize(m);
  auto v0_it = v0->GetData().begin(), v1_it = v1->GetData().begin();
  if(off0 == 0.0 && off1 == 0.0){
    for(size_t i=0; i<m; (i++, v0_it+=stride, v1_it+=stride))
      out[i] = inner_product(v0_it, v0_it+stride, v1_it, 0.0);
  }
  else{
    for(size_t i=0; i<n; i++)
      out[i/stride] += (v0->GetVal(i) + off0) * (v1->GetVal(i) + off1);
  }
  if(snorm){
    double norm = sqrt(stride);
    transform(out.begin(), out.end(), out.begin(),
	      [norm](double val){ return val/norm; });
  }
}


void GHAModel::StridedInnerProduct(PMatrix<double>* v0, PMatrix<double>* v1,
				   const vector<size_t>& stride,
				   vector<double>& out,
				   double off0, double off1, bool snorm){
  assert(v0->GetNData() == v1->GetNData());
  assert(accumulate(stride.begin(), stride.end(), 0) == (int) v0->GetNData());
  out.resize(stride.size());
  fill(out.begin(), out.end(), 0.0);
  auto v0_it = v0->GetData().begin(), v1_it = v1->GetData().begin();
  if(off0 == 0.0 && off1 == 0.0){
    for(size_t s=0; s<stride.size(); s++)
      for(size_t i=0; i<stride[s]; (i++, v0_it++, v1_it++))
	out[s] += (*v0_it) * (*v1_it);
  }
  else{
    for(size_t s=0; s<stride.size(); s++)
      for(size_t i=0; i<stride[s]; (i++, v0_it++, v1_it++))
	out[s] += (*v0_it + off0) * (*v1_it + off1);
  }
  if(snorm)
    for(size_t i=0; i<out.size(); i++) out[i] /= sqrt(stride[i]);
}

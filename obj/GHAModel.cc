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
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <algorithm>
#ifdef __CUDA__
#include "CUDAUtils.hh"
#include "gha_kernels.h"
#endif

using namespace std;
using namespace std::chrono;

// Format console diagnostics without changing floating-point calculations or
// stream state. Four digits after the decimal; very small values use scientific
// notation so convergence residuals and GWAS p-values are not rounded to zero.
static std::string Log4(double value){
  std::ostringstream out;
  if(std::isfinite(value) && value != 0.0 && std::fabs(value) < 1.e-4)
    out << std::scientific;
  else
    out << std::fixed;
  out << std::setprecision(4) << value;
  return out.str();
}

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
  use_var_ratio    = false;
  vr_weights_ready = false;
  vr_maf_edges     = vector<double>({0.001, 0.5});
  InitVarRatioBins();
  spa_thresh    = numeric_limits<double>::max();
  // Robust, MAF-stratified VR calibration defaults.
  vr_init_eval      = 30;
  vr_step_eval      = 10;
  vr_max_eval       = 500;
  vr_stable_checks  = 2;
  vr_epsilon        = 0.01;  // 1% consecutive-median stability threshold
  vr_diagnostics    = false;
  ap_audit = false;
  ap_tau_seed = 0.5;
  fixed_tau_mode = false;
  fixed_tau_value = 0.5;
  vr_audit_enabled = false;
  ap_scores_fresh = false; // -S may load cached info without Step 1 scores
  tau_fd_step = 0.0;
  tau_fd_ready = false;
  tau_fd_cross_enabled = false;
  tau_fd_fast_w1 = false;
  hq_fd_audit_enabled = false;
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
  cout << "  " << Log4(duration_cast<nanoseconds>(t3-t2).count()*1.0e-9) << " s" << endl
       << "inverting pairwise psi..." << endl;
  psiinv = new BDMatrix<double>("psiinv", *psi);
  psiinv->Invert();
  t2 = high_resolution_clock::now();
  cout << "  " << Log4(duration_cast<nanoseconds>(t2-t3).count()*1.0e-9)
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
  XtW_resid       = NULL;
  XtWX_inv_resid  = NULL;
  XtWG_resid      = NULL;
  coefG_resid     = NULL;
  G_tilde_in      = NULL;
  residualizer_ready  = false;
  g_projection_ready  = false;
  g_tilde_ready       = false;
  g_working_var      = numeric_limits<double>::quiet_NaN();
  g_resid_coef.clear();
  
  
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
      cout << pr.first << ":  " << Log4(pr.second) << " us  "
	   << Log4(pr.second/niterations) << " us/iter" << endl;
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
  if(XtW_resid)      delete XtW_resid;
  if(XtWX_inv_resid) delete XtWX_inv_resid;
  if(XtWG_resid)     delete XtWG_resid;
  if(coefG_resid)    delete coefG_resid;
  if(G_tilde_in)     delete G_tilde_in;
  
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


// ============================================================
// variance ratio MAF bins
// ============================================================

void GHAModel::InitVarRatioBins(){
  if(vr_maf_edges.size() < 2)
    vr_maf_edges = vector<double>({0.001, 0.5});
  const size_t n = vr_maf_edges.size() - 1;
  var_ratio.assign(n, vector<double>());
  var_ratio_found.assign(n, false);
  var_ratio_user.assign(n, false);
  vr_mean.assign(n, 0.0);
  vr_prev.assign(n, numeric_limits<double>::quiet_NaN());
  vr_stable_count.assign(n, 0);
  vr_bin = 0;
}


void GHAModel::SetVarRatioMAFEdges(const vector<double>& edges){
  if(edges.size() < 2){
    cerr << "SetVarRatioMAFEdges: need at least two edges, ignoring." << endl;
    return;
  }
  vr_maf_edges = edges;
  sort(vr_maf_edges.begin(), vr_maf_edges.end());
  InitVarRatioBins();   // clears every per-bin value
}


// MAF falls in bin i when it satisfies maf <= vr_maf_edges[i+1].
// anything below the first edge is folded into bin 0.
size_t GHAModel::GetVarRatioBin(double maf) const{
  const size_t n = vr_mean.size();
  if(n <= 1) return 0;
  for(size_t i=0; i+1<vr_maf_edges.size(); i++)
    if(maf <= vr_maf_edges[i+1]) return std::min(i, n-1);
  return n - 1;
}


bool GHAModel::AllVarRatioFound() const{
  for(size_t i=0; i<var_ratio_found.size(); i++)
    if(!var_ratio_found[i]) return false;
  return true;
}


// every bin's exact evaluations concatenated.  the returned reference is
// only valid until the next call.  bin membership is lost, so this is for
// the legacy HDF5 writers only, never for computing a median.
const vector<double>& GHAModel::GetVarRatioEstimates() const{
  size_t n = 0;
  for(size_t b=0; b<var_ratio.size(); b++) n += var_ratio[b].size();
  var_ratio_all.clear();
  var_ratio_all.reserve(n);
  for(size_t b=0; b<var_ratio.size(); b++)
    var_ratio_all.insert(var_ratio_all.end(),
			 var_ratio[b].begin(), var_ratio[b].end());
  return var_ratio_all;
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

void GHAModel::PrepareGResidualizer(){

  if(residualizer_ready)
    return;

  // ----------------------------------------------------------
  // Number of nuisance covariates + intercept
  // ----------------------------------------------------------

  const size_t q =
      X_in->GetNCols() + 1;

  // ----------------------------------------------------------
  // Final working weights
  //
  // IMPORTANT:
  // use exactly the same W as ComputeVarRatio / ComputeCGFs
  //
  // W_i = clambda_i * ebxw1_i
  // ----------------------------------------------------------

  for(size_t i=0; i<nind; i++){

    ub_in->SetVal(
        i,
        clambda_in->GetVal(i) *
        ebxw1_in->GetVal(i)
    );
  }

  vr_weights_ready = true;


  // ----------------------------------------------------------
  // XtW_resid = Xaug' W
  //
  // dimension:
  // q x n
  //
  // row 0 = intercept
  // row 1... = original covariates
  // ----------------------------------------------------------

  if(XtW_resid) delete XtW_resid;

  XtW_resid =
      new PMatrix<double>(
          "XtW_resid",
          q,
          nind
      );

  XtW_resid->SetUseCblas(true);


  for(size_t i=0; i<nind; i++){

    const double wi =
        ub_in->GetVal(i);

    // intercept
    XtW_resid->SetVal(
        0,
        i,
        wi
    );

    for(size_t j=0; j<X_in->GetNCols(); j++){

      XtW_resid->SetVal(
          j + 1,
          i,
          wi *
          X_in->GetVal(i, j)
      );
    }
  }


  // ----------------------------------------------------------
  // A = Xaug' W Xaug
  // ----------------------------------------------------------

  PMatrix<double>* A =
      new PMatrix<double>(
          "XtWX_resid",
          q,
          q
      );

  A->Zero();

  for(size_t i=0; i<nind; i++){

    const double wi =
        ub_in->GetVal(i);

    for(size_t j=0; j<q; j++){

      const double xj =
          (j == 0)
          ?
          1.0
          :
          X_in->GetVal(
              i,
              j - 1
          );

      for(size_t k=j; k<q; k++){

        const double xk =
            (k == 0)
            ?
            1.0
            :
            X_in->GetVal(
                i,
                k - 1
            );

        A->SetVal(
            j,
            k,
            A->GetVal(j, k) +
            wi * xj * xk
        );
      }
    }
  }

  A->MakeSymmetric(true);


  // ----------------------------------------------------------
  // A^{-1}
  // ----------------------------------------------------------

  if(XtWX_inv_resid) delete XtWX_inv_resid;

  XtWX_inv_resid =
      A->Inverse();

  delete A;


  // ----------------------------------------------------------
  // Reusable small vectors
  // ----------------------------------------------------------

  if(XtWG_resid) delete XtWG_resid;

  XtWG_resid =
      new PMatrix<double>(
          "XtWG_resid",
          q,
          1
      );

  if(coefG_resid) delete coefG_resid;

  coefG_resid =
      new PMatrix<double>(
          "coefG_resid",
          q,
          1
      );

  if(G_tilde_in) delete G_tilde_in;

  G_tilde_in =
      new PMatrix<double>(
          "G_tilde_in",
          nind,
          1
      );

  g_resid_coef.assign(q, 0.0);


  residualizer_ready =
      true;

  g_projection_ready =
      false;

  g_tilde_ready =
      false;
  g_working_var = numeric_limits<double>::quiet_NaN();


  // Residualizer initialized; no routine console output is needed.
}

double GHAModel::ComputeResidualizedWorkingVar(){
  // Residualize G_c with respect to [1, X] under the SAME null-model
  // working weights used for the variance-ratio denominator and SPA.
  // This is the single source of truth for the score-test genotype.
  PrepareGResidualizer();
  const size_t q = X_in->GetNCols() + 1;
  vector<double> xtwg(q, 0.0);
  for(size_t i=0; i<nind; ++i){
    const double weighted_g = ub_in->GetVal(i) * G_in->GetVal(i);
    xtwg[0] += weighted_g;
    for(size_t a=1; a<q; ++a)
      xtwg[a] += X_in->GetVal(i, a-1) * weighted_g;
  }
  g_resid_coef.assign(q, 0.0);
  for(size_t a=0; a<q; ++a){
    for(size_t b=0; b<q; ++b)
      g_resid_coef[a] += XtWX_inv_resid->GetVal(a,b) * xtwg[b];
    if(!std::isfinite(g_resid_coef[a]))
      throw std::runtime_error("Non-finite weighted genotype residualization coefficient");
  }

  // Direct weighted norm avoids catastrophic cancellation in
  // G_c' W G_c - (X' W G_c)' (X' W X)^{-1} (X' W G_c).
  double working_var = 0.0;
  for(size_t i=0; i<nind; ++i){
    double fitted = g_resid_coef[0];
    for(size_t a=1; a<q; ++a)
      fitted += X_in->GetVal(i,a-1) * g_resid_coef[a];
    const double gtilde = G_in->GetVal(i) - fitted;
    const double wi = ub_in->GetVal(i);
    if(!std::isfinite(gtilde) || !std::isfinite(wi) || wi < 0.0)
      throw std::runtime_error("Invalid working weight or residualized genotype");
    G_tilde_in->SetVal(i, gtilde);
    working_var += wi * gtilde * gtilde;
  }
  g_projection_ready = true;
  g_tilde_ready = true;
  g_working_var = working_var;
  return working_var;
}

void GHAModel::BuildResidualizedG(){
  if(!g_tilde_ready) (void)ComputeResidualizedWorkingVar();
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
  for(size_t i=0; i<expbx->GetNData(); i++)
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
  for(size_t i=0; i<var_ratio_found.size(); i++) var_ratio_found[i] = false;
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
          cout << "INNER converged: beta/lambda\n";
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
      cout << "OUTER " << outer+1
           << " tau_before=" << Log4(tauprev)
           << " tau=" << Log4(tau)
           << " |dTau|=" << Log4(dTau)
           << " time(us)=" << Log4(duration_cast<nanoseconds>(t1_outer-t0_outer).count()*1.0e-3)
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
         << "||beta|| = "   << Log4(bnorm) << ", "
         << "||lambda|| = " << Log4(lnorm) << ", "
         << "|tau| = "      << Log4(tnorm) << ", "
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
	 << "||beta|| = "   << Log4(bnorm) << ", "
	 << "||lambda|| = " << Log4(lnorm) << ", "
	 << "|tau| = "      << Log4(tnorm) << endl;
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
    sum = 0.0; // Each hazard score has its own risk set; do not cumulate across event times.
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
  ap_scores_fresh = true;
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
  // A refitted null model invalidates the previous calibration samples and
  // convergence history. User-supplied ratios are preserved separately.
  for(size_t i=0; i<var_ratio_found.size(); i++){
    var_ratio_found[i] = var_ratio_user[i];
    var_ratio[i].clear();
    vr_prev[i] = numeric_limits<double>::quiet_NaN();
    vr_stable_count[i] = 0;
    if(!var_ratio_user[i]) vr_mean[i] = 0.0;
  }
  // the null model is about to be refit, so the cached working weights
  // and the residualizer built from them are no longer valid
  vr_weights_ready   = false;
  residualizer_ready = false;
  g_projection_ready = false;
  g_tilde_ready      = false;
  g_working_var     = numeric_limits<double>::quiet_NaN();
  TimeSortData("TimeSortDataIn", T_in_perm, T_in_perm_t,
	       T_in, T_in_sort, X_in, X_in_sort, delta_in, delta_in_sort,
	       ebxw1_in, ebxw1_in_sort);
  double dsum = accumulate(delta_in_sort->GetDataConst().begin(),
			   delta_in_sort->GetDataConst().end(), 0.0);
  for(size_t i=0; i<lambda_in->GetNData(); i++){
    lambda_in->SetVal(i, delta_in_sort->GetVal(i) / dsum);
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
  // ub_in no longer holds the final working weights
  vr_weights_ready = false;
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
      cout << "iteration " << niterations << " "
	   << Log4(duration_cast<nanoseconds>(t1-t0).count()*1.0e-3) << " us:" <<endl;
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
	 << "||beta|| = "   << Log4(bnorm) << ", "
	 << "||lambda|| = " << Log4(lnorm) << endl;
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
			 clambda_in->GetVal(i) * ebxw1_in->GetVal(i), *X_in);
  auto t1 = high_resolution_clock::now();
  titer_ist_us["ScoreIn_beta"]+=duration_cast<nanoseconds>(t1-t0).count()*1e-3;
}


void GHAModel::ScoreIn_h(){
  auto t0 = high_resolution_clock::now();
  if(Uh_unq_in) Uh_unq_in->Zero();
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
  if(vr_audit_enabled){
    // Three deterministic column probes for ||I - V*Vinv||_infty.
    // This is NOT a condition number; only a numerical-inverse check.
    const size_t nprobe = V_unq_in->GetNRows();
    double inverse_error = 0.0;
    // Since Vinv was already computed, the induced infinity-norm
    // condition estimate costs only O(n^2) additional arithmetic.
    double information_inf_norm = 0.0;
    double inverse_inf_norm = 0.0;
    double diagonal_min = std::numeric_limits<double>::infinity();
    double diagonal_max = 0.0;
    for(size_t z=0; z<nprobe; ++z){
      double row_sum = 0.0, inverse_row_sum = 0.0;
      for(size_t col=0; col<nprobe; ++col){
        row_sum += std::fabs(V_unq_in->GetVal(z,col));
        inverse_row_sum += std::fabs(Vinv_unq_in->GetVal(z,col));
      }
      information_inf_norm = std::max(information_inf_norm, row_sum);
      inverse_inf_norm = std::max(inverse_inf_norm, inverse_row_sum);
      const double diag = std::fabs(V_unq_in->GetVal(z,z));
      if(diag > 0.0 && std::isfinite(diag))
        diagonal_min = std::min(diagonal_min,diag);
      if(std::isfinite(diag)) diagonal_max=std::max(diagonal_max,diag);
    }
    std::vector<size_t> cols;
    if(nprobe > 0) cols.push_back(0);
    if(nprobe > 2) cols.push_back(nprobe/2);
    if(nprobe > 1) cols.push_back(nprobe-1);
    for(size_t col : cols){
      for(size_t row=0; row<nprobe; ++row){
        double val=0.0;
        for(size_t k=0; k<nprobe; ++k)
          val+=V_unq_in->GetVal(row,k)*Vinv_unq_in->GetVal(k,col);
        val -= (row==col ? 1.0 : 0.0);
        inverse_error=std::max(inverse_error,std::fabs(val));
      }
    }
    cout << "[INFORMATION_IN_AUDIT] inverse_identity_max_abs="
         << inverse_error << " cond_inf_est="
         << information_inf_norm * inverse_inf_norm
         << " norm_inf=" << information_inf_norm
         << " inverse_norm_inf=" << inverse_inf_norm
         << " diag_abs_ratio="
         << (diagonal_min > 0.0 ? diagonal_max/diagonal_min
                                : std::numeric_limits<double>::infinity())
         << " note=cond_inf_uses_computed_inverse" << endl;
  }
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


// FD-only posterior mean kernel.  Implements EXACTLY the w1 formulas from
// ComputeIn_EV_GH (small blocks) and ComputeIn_EV (larger blocks), but does
// NOT compute Ebb/Eebb/Ebbeb or touch the fitted null parameters.
// This kernel is opt-in and is never used during fitting or association score.
void GHAModel::ComputeIn_EV_W1Only(){
  w1_in->Zero();
  const double stau=std::sqrt(tau);
  GaussHermiteND gh1=gauss_hermite.back();
  for(size_t q=0;q<psiinv_in->GetNBlocks();++q){
    const size_t n=psiinv_in->GetBlockRows(q);
    const size_t offset=psiinv_in->GetBlockOffset(q);
    if(n==0 || offset+n>nind)
      throw std::runtime_error("FD W1-only: invalid original block size or offset");
    if(n<=gauss_hermite.size()){
      // Keep the same full-block quadrature as ComputeIn_EV_GH(q).
      const size_t g=gauss_hermite.size()-n;
      GaussHermiteND gh=gauss_hermite[g];
      const size_t p=gh.GetPowers()[0];
      PMatrix<double>* block=psi_in->GetBlock(q);
      PMatrix<double> sqrtpsi("fd_sqrt_psi",n,n);
      block->Sqrt(&sqrtpsi);
      delete block;
      PMatrix<double> zz("fd_gh_z",n,1);
      PMatrix<double> bb("fd_gh_b",n,1);
      std::vector<double> roots;
      std::vector<double> numerator(n,0.0);
      double denominator=0.0;
      for(size_t j=0;j<p;++j){
        gh.GetRootSqrt2Permutation(roots,j);
        for(size_t k=0;k<n;++k) zz.SetVal(k,roots[k]);
        sqrtpsi.RightMult(zz,&bb);
        bb *= stau;
        double log_event=0.0,log_hazard=0.0;
        for(size_t k=0;k<n;++k){
          const size_t i=offset+k;
          const double earg=std::exp(bb.GetVal(k));
          zz.SetVal(k,earg);
          log_event += delta_in->GetVal(i)*bb.GetVal(k);
          log_hazard += clambda_in->GetVal(i)*ebx_in->GetVal(i)*earg;
        }
        const double weight=gh.GetWeightProd(j)*std::exp(log_event-log_hazard);
        denominator+=weight;
        for(size_t k=0;k<n;++k)
          numerator[k]+=zz.GetVal(k)*weight;
      }
      if(!std::isfinite(denominator) || denominator<=0.0)
        throw std::runtime_error("FD W1-only: invalid GH block normalization");
      for(size_t k=0;k<n;++k){
        const double v=numerator[k]/denominator;
        if(!std::isfinite(v))
          throw std::runtime_error("FD W1-only: nonfinite small-block posterior mean");
        w1_in->SetVal(offset+k,v);
      }
    }else{
      // Exactly the 1D posterior-mean marginal approximation in ComputeIn_EV.
      for(size_t k=0;k<n;++k){
        const size_t i=offset+k;
        const double mui=mu_in->GetVal(i);
        const double sigmai=sigma_in->GetVal(i);
        double denominator=0.0,numerator=0.0;
        for(size_t j=0;j<gh1.GetOrder();++j){
          double arg=sigmai*gh1.GetRootSqrt2(j)+mui;
          double m=delta_in->GetVal(i)*arg;
          arg=std::exp(arg);
          m-=clambda_in->GetVal(i)*ebx_in->GetVal(i)*arg;
          m=std::exp(m)*gh1.GetWeight(j);
          denominator+=m;
          numerator+=m*arg;
        }
        const double v=(numerator==0.0) ? numerator : numerator/denominator;
        if(!std::isfinite(v))
          throw std::runtime_error("FD W1-only: nonfinite large-block posterior mean");
        w1_in->SetVal(i,v);
      }
    }
  }
}

// FIXED-NUISANCE NUMERICAL TAU DERIVATIVE. The nuisance parameters beta and
// Breslow hazard jumps are held fixed. Only the random-effect conditional mode
// and the existing approximate posterior moments are recomputed when tau is
// changed. This does NOT change the fitted null model or genetic score.
void GHAModel::PrepareTauFD(const string& prefix){
  if(!(std::isfinite(tau_fd_step) && tau_fd_step>0.0 &&
       tau_fd_step<tau))
    throw std::runtime_error("Tau FD step must satisfy 0 < step < fitted tau");
  if(!w1_in || !b_in || !Ebb_in || !Ebbeb_in || !V_unq_in || !Vinv_unq_in)
    throw std::runtime_error("Tau FD needs loaded/computed Step 2 score and information");
  // A prefix is optional when the FD Jacobian is needed for VR calibration
  // rather than for an audit. Empty => calculate but do not write TSV files.

  // Keep routine FD logs quiet even when generic -v is used. Detailed FD
  // progress is intended only for explicit numerical audit runs that supply
  // --tau-fd-audit-prefix AND -v. This affects logging only.
  const bool fd_detail = verbose && !prefix.empty();
  const auto fd_wall_start = high_resolution_clock::now();
  if(fd_detail) std::cout << "[TAU_FD_TIMING] begin: n_ind=" << nind
                         << " n_event_times=" << (T_unq ? T_unq->GetNData() : 0)
                         << std::endl;
  const double tau_saved = tau;
  vector<double> b_saved = b_in->GetDataConst();
  vector<double> bprev_saved = bprev_in->GetDataConst();
  vector<double> w1_saved = w1_in->GetDataConst();
  // Perturbed W1-only calls leave Ebb/Eebb/Ebbeb untouched.
  if(fd_detail) std::cout << "[GADO_TAU_CROSS] derivative=central_fd; posterior="
                         << (tau_fd_fast_w1 ? "FIRST_MOMENT_ONLY" : "FULL_POSTERIOR_MOMENTS") << std::endl;
  if(w1_saved.size()!=nind || b_saved.size()!=nind)
    throw std::runtime_error("Tau FD baseline moment dimensions inconsistent");

  PMatrix<double> gradbuff("tau_fd_gradbuf", nind, 1);
  auto mode_residual=[&]() -> double {
    psiinv_in->RightMult(*b_in, &gradbuff);
    double maxres=0.0;
    for(size_t i=0;i<nind;++i){
      const double res = delta_in->GetVal(i) -
        clambda_in->GetVal(i)*ebx_in->GetVal(i)*std::exp(b_in->GetVal(i)) -
        gradbuff.GetVal(i)/tau;
      maxres=std::max(maxres,std::fabs(res));
    }
    return maxres;
  };
  const double base_mode_residual=mode_residual();
  vector<double> wm(nind),wp(nind);
  double side_mode_residual[2]={0.0,0.0};
  int side_steps[2]={0,0};
  bool side_converged[2]={false,false};
  for(int side=0;side<2;++side){
    const auto side_start=high_resolution_clock::now();
    const char* side_name=(side==0 ? "minus" : "plus");
    if(fd_detail) std::cout << "[TAU_FD_STAGE] side=" << side_name
                          << " starting conditional-mode Newton" << std::endl;
    tau=tau_saved + (side==0 ? -tau_fd_step : tau_fd_step);
    b_in->SetData(b_saved);
    // Newton updates ONLY to the conditional mode at the perturbed tau.
    // We do not call UpdateBeta, UpdateLambda, or IterateIn.
    for(int k=0;k<25;++k){
      UpdateIn_b();
      ++side_steps[side];
      double maxstep=0.0;
      for(size_t i=0;i<nind;++i)
        maxstep=std::max(maxstep, std::fabs(
          b_in->GetVal(i)-bprev_in->GetVal(i)));
      if(fd_detail && (k==0 || (k+1)%5==0))
        std::cout << "[TAU_FD_STAGE] side=" << side_name
                  << " newton_steps=" << k+1 << " max_step=" << Log4(maxstep)
                  << " elapsed_s=" << Log4(duration_cast<nanoseconds>(high_resolution_clock::now()-side_start).count()*1e-9) << std::endl;
      if(maxstep<1.e-10){ side_converged[side]=true; break; }
    }
    side_mode_residual[side]=mode_residual();
    if(!side_converged[side] || side_mode_residual[side]>1.e-5)
      throw std::runtime_error("Tau FD conditional b mode failed to converge at perturbed tau");
    if(fd_detail) std::cout << "[TAU_FD_STAGE] side=" << side_name
              << " newton_done steps=" << side_steps[side]
              << " residual=" << Log4(side_mode_residual[side])
              << " elapsed_s=" << Log4(duration_cast<nanoseconds>(high_resolution_clock::now()-side_start).count()*1e-9) << std::endl;
    UpdateIn_mus();
    if(tau_fd_fast_w1) ComputeIn_EV_W1Only();
    else ComputeIn_EV();
    if(fd_detail) std::cout << "[TAU_FD_STAGE] side=" << side_name << " posterior_done"
              << " elapsed_s=" << Log4(duration_cast<nanoseconds>(high_resolution_clock::now()-side_start).count()*1e-9) << std::endl;
    vector<double>& target=(side==0 ? wm : wp);
    for(size_t i=0;i<nind;++i){
      target[i]=w1_in->GetVal(i);
      if(!std::isfinite(target[i]))
        throw std::runtime_error("Tau FD nonfinite posterior moment");
    }
  }

  if(fd_detail) std::cout << "[TAU_FD_TIMING] two-sided mode/moment calculations: "
            << Log4(duration_cast<nanoseconds>(high_resolution_clock::now()-fd_wall_start).count()*1e-9)
            << " s" << std::endl;
  // Restore the identical fitted tau and conditional mode. In fast mode
  // no high posterior moments were modified; restore stored baseline W1/mu/
  // sigma, and verify the fixed-tau posterior mean matches the cached center.
  // Legacy mode retains its original full center recomputation.
  tau=tau_saved;
  b_in->SetData(b_saved);
  bprev_in->SetData(bprev_saved);
  UpdateIn_mus();
  if(tau_fd_fast_w1){
    ComputeIn_EV_W1Only();
    double center_gap=0.0;
    for(size_t i=0;i<nind;++i)
      center_gap=std::max(center_gap,std::fabs(w1_in->GetVal(i)-w1_saved[i]));
    if(fd_detail) std::cout << "[TAU_FD_STAGE] center_w1_consistency_max_abs=" << Log4(center_gap) << std::endl;
    if(!std::isfinite(center_gap) || center_gap>1e-7)
      throw std::runtime_error("Fast FD center W1 differs from cached Step2 W1; use legacy mode and check -S consistency");
    w1_in->SetData(w1_saved);
  }else{
    ComputeIn_EV();
  }
  UpdateIn_ebxw1();
  double restore_max=0.0, w_fd_l2=0.0, w_gap_l2=0.0, w_analytic_l2=0.0;
  tau_fd_dw1.assign(nind,0.0);
  tau_fd_direct_coef.assign(nind,0.0);
  vector<double> w_deriv_analytic(nind,0.0);
  vector<size_t> tau_fd_original_block(nind,0), tau_fd_block_size(nind,0);
  double w_gap_l2_small=0.0, w_gap_l2_large=0.0;
  size_t n_small=0,n_large=0;
  for(size_t i=0;i<nind;++i)
    restore_max=std::max(restore_max,std::fabs(w1_in->GetVal(i)-w1_saved[i]));
  if(restore_max>1.e-7)
    throw std::runtime_error("Tau FD restoration changed baseline posterior moments");

  const size_t L=beta->GetNData();
  const size_t M=T_unq->GetNData();
  const size_t N=L+M;
  tau_fd_c.assign(N,0.0);
  PMatrix<double> p0;
  PMatrix<double> p1;
  for(size_t b=0;b<psiinv_in->GetNBlocks();++b){
    psiinv_in->GetBlock(b,p0);
    Ebb_in->GetBlock(b,p1);
    const double ebquad=p0.TraceProductSymm(p1);
    const size_t off=psiinv_in->GetBlockOffset(b);
    const size_t size=psiinv_in->GetBlockRows(b);
    for(size_t k=0;k<size;++k){
      const size_t i=off+k;
      tau_fd_original_block[i]=b;
      tau_fd_block_size[i]=size;
      const double num=(wp[i]-wm[i])/(2.0*tau_fd_step);
      const double analytic=(Ebbeb_in->GetVal(i) - w1_in->GetVal(i)*ebquad)/
                            (2.0*tau_saved*tau_saved);
      tau_fd_dw1[i]=num;
      w_deriv_analytic[i]=analytic;
      tau_fd_direct_coef[i]=clambda_in->GetVal(i)*ebx_in->GetVal(i)*num;
      w_fd_l2+=num*num;
      w_gap_l2+=(num-analytic)*(num-analytic);
      if(size<=2){w_gap_l2_small+=(num-analytic)*(num-analytic); ++n_small;}
      else {w_gap_l2_large+=(num-analytic)*(num-analytic); ++n_large;}
      w_analytic_l2+=analytic*analytic;
      for(size_t j=0;j<L;++j)
        tau_fd_c[j]+=tau_fd_direct_coef[i]*X_in->GetVal(i,j);
    }
  }
  // Baseline-jump score: U_hj = d_j/h_j - sum_{T_i>=t_j} exp(x_i beta) E(e^bi).
  // Hence c_fd,j = -dU_hj/dtau = sum_{T_i>=t_j} exp(x_i beta) dw1_i/dtau.
  //
  // The previous nested loop scanned every subject for every event time:
  // O(nind*M). We already have T_in_sort in descending time order and its
  // permutation T_in_perm from TimeSortData("TimeSortDataIn",...). T_unq is
  // descending, too. Sweep each subject once, with inclusive >= ties.
  // This changes only the summation order, not the estimator.
  const auto fd_riskset_start = high_resolution_clock::now();
  if(T_in_perm.size() != nind || T_in_sort->GetNData() != nind)
    throw std::runtime_error("Tau FD riskset: sorted subject/permutation dimensions mismatch");
  size_t cursor=0;
  double risk_sum=0.0;
  for(size_t j=0;j<M;++j){
    const double tj=T_unq->GetVal(j);
    if(j>0 && tj>T_unq->GetVal(j-1))
      throw std::runtime_error("Tau FD riskset: event-time vector is not descending");
    while(cursor<nind && T_in_sort->GetVal(cursor)>=tj){
      const size_t original=T_in_perm[cursor];
      if(original>=nind)
        throw std::runtime_error("Tau FD riskset: invalid sorted subject index");
      risk_sum += ebx_in->GetVal(original)*tau_fd_dw1[original];
      ++cursor;
    }
    tau_fd_c[L+j]=risk_sum;
  }
  if(fd_detail) std::cout << "[TAU_FD_TIMING] riskset cross-information: "
                         << Log4(duration_cast<nanoseconds>(high_resolution_clock::now()-fd_riskset_start).count()*1e-9)
                         << " s (linear sweep, n_ind=" << nind << ", n_event_times=" << M << ")"
                         << std::endl;

  double max_c_beta=0.,max_c_h=0.,c_analytic_l2=0.,c_fd_l2=0.;
  if(!prefix.empty()){
    std::ofstream nuisance((prefix+"_nuisance_fd.tsv").c_str());
    std::ofstream moments((prefix+"_moments_fd.tsv").c_str());
    std::ofstream summary((prefix+"_summary.tsv").c_str());
    if(!nuisance || !moments || !summary)
      throw std::runtime_error("Cannot open tau FD output files for prefix: "+prefix);
    nuisance<<std::setprecision(17);
    moments<<std::setprecision(17);
    summary<<std::setprecision(17);
    nuisance<<"parameter\tindex\tI_xi_tau_analytic\tI_xi_tau_fd\tgap\n";
    for(size_t j=0;j<N;++j){
      const double val=V_unq_in->GetVal(j,N);
      const double gap=tau_fd_c[j]-val;
      nuisance<<(j<L ? "beta" : "h")<<'\t'<<(j<L ? j : j-L)<<'\t'
              <<val<<'\t'<<tau_fd_c[j]<<'\t'<<gap<<'\n';
      if(j<L) max_c_beta=std::max(max_c_beta,std::fabs(gap));
      else max_c_h=std::max(max_c_h,std::fabs(gap));
      c_analytic_l2+=val*val;
      c_fd_l2+=tau_fd_c[j]*tau_fd_c[j];
    }
    moments<<"subject_index\toriginal_block_id\tblock_size\tw1_baseline\tw1_minus\tw1_plus\tdw1_fd\tdw1_analytic\tdw1_gap\n";
    for(size_t i=0;i<nind;++i){
      moments<<i<<'\t'<<tau_fd_original_block[i]<<'\t'<<tau_fd_block_size[i]<<'\t'<<w1_in->GetVal(i)<<'\t'<<wm[i]<<'\t'<<wp[i]<<'\t'
             <<tau_fd_dw1[i]<<'\t'<<w_deriv_analytic[i]<<'\t'
             <<tau_fd_dw1[i]-w_deriv_analytic[i]<<'\n';
    }
    summary<<"tau\tfd_step\tn_subjects\tn_nuisance\tbaseline_mode_residual"
              "\tminus_mode_residual\tplus_mode_residual\tminus_nr_steps"
              "\tplus_nr_steps\tw1_restore_max_abs\tdw1_fd_l2"
              "\tdw1_analytic_l2\tdw1_difference_l2\tI_xi_tau_beta_gap_max"
              "\tI_xi_tau_h_gap_max\tI_xi_tau_analytic_l2\tI_xi_tau_fd_l2"
              "\tn_subjects_block_le2\tn_subjects_block_gt2"
              "\tdw1_gap_l2_block_le2\tdw1_gap_l2_block_gt2\n";
    summary<<tau_saved<<'\t'<<tau_fd_step<<'\t'<<nind<<'\t'<<N<<'\t'
           <<base_mode_residual<<'\t'<<side_mode_residual[0]<<'\t'
           <<side_mode_residual[1]<<'\t'<<side_steps[0]<<'\t'<<side_steps[1]
           <<'\t'<<restore_max<<'\t'<<std::sqrt(w_fd_l2)<<'\t'
           <<std::sqrt(w_analytic_l2)<<'\t'<<std::sqrt(w_gap_l2)
           <<'\t'<<max_c_beta<<'\t'<<max_c_h<<'\t'
           <<std::sqrt(c_analytic_l2)<<'\t'<<std::sqrt(c_fd_l2)<<'\t'
           <<n_small<<'\t'<<n_large<<'\t'<<std::sqrt(w_gap_l2_small)<<'\t'
           <<std::sqrt(w_gap_l2_large)<<'\n';
    if(fd_detail) std::cout << "[TAU_FD_AUDIT] tau=" << Log4(tau_saved)
                          << " step=" << Log4(tau_fd_step)
                          << " base_mode_residual=" << Log4(base_mode_residual)
                          << " w1_restore_error=" << Log4(restore_max)
                          << " dw1_analytic_fd_l2_gap=" << Log4(std::sqrt(w_gap_l2))
                          << " max_beta_c_gap=" << Log4(max_c_beta)
                          << " max_h_c_gap=" << Log4(max_c_h) << std::endl;
    std::cout<<"[TAU_FD] output "<<prefix<<"_{summary,nuisance_fd,moments_fd}.tsv"
             <<std::endl;
  }else if(fd_detail){
    std::cout << "[TAU_FD_AUDIT] w1_restore_error=" << Log4(restore_max)
              << " dw1_analytic_fd_l2_gap=" << Log4(std::sqrt(w_gap_l2))
              << std::endl;
  }
  // H and q remain analytic. Cross-information is stored separately and
  // subsequently used in the exact block-sandwich variance when requested.
  vr_weights_ready=false;
  residualizer_ready=false;
  g_projection_ready=false;
  g_tilde_ready=false;
  tau_fd_ready=true;
  std::cout << "[GADO_TAU_CROSS] prepared: tau=" << Log4(tau_saved)
            << " step=" << Log4(tau_fd_step)
            << " elapsed_s=" << Log4(duration_cast<nanoseconds>(high_resolution_clock::now()-fd_wall_start).count()*1e-9)
            << std::endl;
  if(fd_detail)
    std::cout << "[TAU_FD_AUDIT] Newton(-/+)=" << side_steps[0] << "/" << side_steps[1]
              << " max_mode_residual=" << Log4(std::max(side_mode_residual[0],side_mode_residual[1]))
              << " w1_restore_error=" << Log4(restore_max) << std::endl;
}

// Validation-only H/q directional Jacobian audit for the *implemented* Step 2
// approximate estimating equations. Tau and the centered genotype used by the exact score stay fixed.
// No production score/information or fitted parameters are overwritten on disk.
// Because the nuisance baseline hazard has many jumps, use directional probes
// (two beta coordinates, proportional hazard-jump scaling, mixed direction).
// This checks a few directions, not the full large Jacobian or asymptotic proof.
void GHAModel::RunHQFDAudit(const string& prefix,double eps){
#ifndef __CUDA__
  if(!hq_fd_audit_enabled || prefix.empty())
    throw std::runtime_error("Hq FD audit not enabled");
  if(!(std::isfinite(eps) && eps>0.0 && eps<0.1))
    throw std::runtime_error("Invalid Hq FD half-step");
  const size_t L=beta->GetNData();
  const size_t M=T_unq->GetNData();
  const size_t N=L+M;
  if(L==0 || M==0 || hq_analytic_q.size()!=N ||
     !V_unq_in || !G_in || T_in_perm.size()!=nind)
    throw std::runtime_error("Hq FD audit missing analytic q / Step2 state");
  const std::vector<double> saved_b=b_in->GetDataConst();
  const std::vector<double> saved_mu=mu_in->GetDataConst();
  const std::vector<double> saved_sigma=sigma_in->GetDataConst();
  const std::vector<double> saved_w=w1_in->GetDataConst();
  const std::vector<double> saved_ebx=ebx_in->GetDataConst();
  const std::vector<double> saved_clambda=clambda_in->GetDataConst();
  // All H/q perturbations keep tau and the centered genotype G_in fixed;
  // the conditional mode and W1 are recalculated at each side.
  std::ofstream out(prefix.c_str());
  if(!out) throw std::runtime_error("Cannot write Hq FD audit: "+prefix);
  out << std::setprecision(17);
  out << "direction\tepsilon\tnewton_minus\tnewton_plus"
         "\tH_fd_l2\tH_analytic_l2\tH_gap_l2\tH_rel_l2"
         "\tH_beta_rel_l2\tH_hazard_rel_l2\tq_fd\tq_analytic"
         "\tq_abs_gap\tq_scaled_rel_gap\tstatus_1pct\n";

  std::vector<std::pair<std::string,std::vector<double>>> dirs;
  std::vector<double> v(N,0.0);
  v[0]=1.0; dirs.push_back({"beta_first",v});
  if(L>1){
    std::fill(v.begin(),v.end(),0.0);
    v[L-1]=1.0; dirs.push_back({"beta_last",v});
  }
  std::fill(v.begin(),v.end(),0.0);
  for(size_t j=0;j<M;++j) v[L+j]=h_unq_in->GetVal(j);
  dirs.push_back({"hazard_proportional",v});
  std::fill(v.begin(),v.end(),0.0);
  v[0]=0.5;
  for(size_t j=0;j<M;++j) v[L+j]=0.5*h_unq_in->GetVal(j);
  dirs.push_back({"beta_hazard_joint",v});

  auto restore=[&](){
    // PMatrix::SetData requires a mutable vector argument.
    std::vector<double> x;
    x=saved_b; b_in->SetData(x);
    x=saved_mu; mu_in->SetData(x);
    x=saved_sigma; sigma_in->SetData(x);
    x=saved_w; w1_in->SetData(x);
    x=saved_ebx; ebx_in->SetData(x);
    x=saved_clambda; clambda_in->SetData(x);
    UpdateIn_ebxw1();
    vr_weights_ready=false;
    residualizer_ready=false;
    g_projection_ready=false;
    g_tilde_ready=false;
  };

  size_t failures=0;
  try{
    for(const auto& dir:dirs){
      const auto& vv=dir.second;
      // Every hazard component is scaled by the same relative factor.
      double hazard_scale=0.0;
      for(size_t j=0;j<M;++j){
        const double hj=h_unq_in->GetVal(j);
        if(!(std::isfinite(hj) && hj>0))
          throw std::runtime_error("Hq FD: invalid fitted baseline hazard jump");
        if(std::fabs(vv[L+j])>0.0){
          hazard_scale=vv[L+j]/hj;
          break;
        }
      }
      for(size_t j=0;j<M;++j){
        const double hj=h_unq_in->GetVal(j);
        if(std::fabs(vv[L+j]-hazard_scale*hj)>1e-12*std::max(1.0,std::fabs(hj)))
          throw std::runtime_error("Hq FD: hazard direction must be proportional to jumps");
      }
      std::vector<double> scores[2];
      double gscore[2]={0.0,0.0};
      int newton_count[2]={0,0};
      for(int side=0;side<2;++side){
        const double signed_eps=(side==0 ? -eps : eps);
        const double hazard_multiplier=1.0+signed_eps*hazard_scale;
        if(hazard_multiplier<=0.0)
          throw std::runtime_error("Hq FD hazard perturbation not positive");
        std::vector<double> work=saved_b;
        b_in->SetData(work);
        for(size_t i=0;i<nind;++i){
          double dbeta=0.0;
          for(size_t j=0;j<L;++j) dbeta+=X_in->GetVal(i,j)*vv[j];
          ebx_in->SetVal(i,saved_ebx[i]*std::exp(signed_eps*dbeta));
          clambda_in->SetVal(i,saved_clambda[i]*hazard_multiplier);
        }
        bool converged=false;
        for(int step=0;step<30;++step){
          UpdateIn_b();
          ++newton_count[side];
          double maxstep=0.0;
          for(size_t i=0;i<nind;++i)
            maxstep=std::max(maxstep,std::fabs(b_in->GetVal(i)-bprev_in->GetVal(i)));
          if(maxstep<1e-10){converged=true;break;}
        }
        if(!converged)
          throw std::runtime_error("Hq FD perturbed mode failed 30 Newton iterations");
        UpdateIn_mus();
        ComputeIn_EV_W1Only();
        scores[side].assign(N,0.0);
        gscore[side]=0.0;
        std::vector<double> weighted(nind,0.0);
        for(size_t i=0;i<nind;++i){
          weighted[i]=ebx_in->GetVal(i)*w1_in->GetVal(i);
          const double residual=delta_in->GetVal(i)-clambda_in->GetVal(i)*weighted[i];
          gscore[side]+=G_in->GetVal(i)*residual;
          for(size_t j=0;j<L;++j) scores[side][j]+=X_in->GetVal(i,j)*residual;
        }
        size_t cursor=0;
        double risk=0.0;
        for(size_t j=0;j<M;++j){
          const double tj=T_unq->GetVal(j);
          double events=0.0;
          while(cursor<nind && T_in_sort->GetVal(cursor)>=tj){
            const size_t idx=T_in_perm[cursor];
            risk+=weighted[idx];
            if(T_in_sort->GetVal(cursor)==tj) events+=delta_in->GetVal(idx);
            ++cursor;
          }
          scores[side][L+j]=events/(h_unq_in->GetVal(j)*hazard_multiplier)-risk;
        }
      }
      std::vector<double> hv(N,0.0), fd(N,0.0);
      double sum_ha=0.0,sum_hf=0.0,sum_gap=0.0;
      double beta_gap=0.,beta_scale=0.,haz_gap=0.,haz_scale=0.;
      for(size_t i=0;i<N;++i){
        for(size_t j=0;j<N;++j) hv[i]+=V_unq_in->GetVal(i,j)*vv[j];
        fd[i]=-(scores[1][i]-scores[0][i])/(2.0*eps);
        const double gap=fd[i]-hv[i];
        sum_ha+=hv[i]*hv[i]; sum_hf+=fd[i]*fd[i]; sum_gap+=gap*gap;
        if(i<L){beta_gap+=gap*gap;beta_scale+=std::max(fd[i]*fd[i],hv[i]*hv[i]);}
        else{haz_gap+=gap*gap;haz_scale+=std::max(fd[i]*fd[i],hv[i]*hv[i]);}
      }
      const double qfd=-(gscore[1]-gscore[0])/(2.0*eps);
      double qa=0.; for(size_t j=0;j<N;++j) qa+=hq_analytic_q[j]*vv[j];
      const double hrel=std::sqrt(sum_gap)/std::max(1.0,std::sqrt(std::max(sum_ha,sum_hf)));
      const double brel=std::sqrt(beta_gap)/std::max(1.0,std::sqrt(beta_scale));
      const double jrel=std::sqrt(haz_gap)/std::max(1.0,std::sqrt(haz_scale));
      const double qrel=std::fabs(qfd-qa)/std::max(1.0,std::max(std::fabs(qfd),std::fabs(qa)));
      // Practical screening threshold only; NOT a theorem or final calibration test.
      const bool pass=std::isfinite(hrel)&&std::isfinite(qrel)&&hrel<=0.01&&qrel<=0.01;
      if(!pass) ++failures;
      out<<dir.first<<'\t'<<eps<<'\t'<<newton_count[0]<<'\t'<<newton_count[1]
         <<'\t'<<std::sqrt(sum_hf)<<'\t'<<std::sqrt(sum_ha)
         <<'\t'<<std::sqrt(sum_gap)<<'\t'<<hrel<<'\t'<<brel<<'\t'<<jrel
         <<'\t'<<qfd<<'\t'<<qa<<'\t'<<std::fabs(qfd-qa)
         <<'\t'<<qrel<<'\t'<<(pass?"PASS":"FAIL")<<'\n';
      std::cout << "[GADO_HQ_AUDIT] direction="<<dir.first
                << " H_rel="<<Log4(hrel)<<" q_rel="<<Log4(qrel)
                << " screening="<<(pass?"PASS":"FAIL")<<std::endl;
    }
  }catch(...){restore();throw;}
  restore();
  out.close();
  std::cout << "[GADO_HQ_AUDIT] completed="<<dirs.size()
            << " screening_failures="<<failures<<" output="<<prefix
            << "; not an asymptotic proof"<<std::endl;
  if(failures) std::cout << "[GADO_HQ_AUDIT] WARNING: H/q screening failed; do not certify release" << std::endl;
#else
  throw std::runtime_error("H/q FD Jacobian audit not supported by CUDA build");
#endif
}

void GHAModel::Solve(size_t maxiterap, size_t maxiterin){
  auto t0 = high_resolution_clock::now();
  SetSeedsAP();
  if(fixed_tau_mode){
    if(!std::isfinite(fixed_tau_value) || fixed_tau_value <= 0.0)
      throw std::runtime_error("Experimental fixed tau must be positive/finite");
    tau = fixed_tau_value;
    tauprev = tau;
    tau_iter.assign(1, tau);
    // Step 1 is intentionally NOT estimated. Step 2 will be fitted with
    // this known variance component. Asc must be zero in score variance.
    converged_ap = true;
    cout << "[EXPERIMENT_FIXED_TAU] Step 1 skipped; known tau="
         << Log4(tau) << endl;
  }else{
    if(maxiterap == 0) converged_ap = IterateAP(SIZE_MAX);
    else converged_ap = IterateAP(maxiterap);
  }
  auto t1 = high_resolution_clock::now();
  cout << endl << "Step 1 solution after " << niterations << " iterations, "
       << Log4(duration_cast<nanoseconds>(t1-t0).count()*1.0e-9) << " s:" << endl;
  beta->Print(beta->GetNData(), 4, "    ");
  cout << "tau = " << Log4(tau) << endl << endl;
  for(auto& pr : titer_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << Log4(pr.second) << " us  "
	   << Log4(pr.second/std::max<size_t>(1,niterations)) << " us/iter" << endl;
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
       << Log4(duration_cast<nanoseconds>(t1-t0).count()*1.0e-9) << " s:" << endl;
  beta->Print(beta->GetNData(), 4, "    ");
  cout << "tau = " << Log4(tau) << endl << endl;
  for(auto& pr : titer_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << Log4(pr.second) << " us  "
	   << Log4(pr.second/std::max<size_t>(1,niterations)) << " us/iter" << endl;
      pr.second = 0.0;
    }
  beta1->SetData(beta->GetData());
  ns_fit_in = duration_cast<nanoseconds>(t1-t0).count();
  niterations_in = niterations;
}



// Diagnostic only: emit Step 1 IF by *original* GRM block. Can be called
// after reading an existing -S score/info file. Cached -S does not guarantee
// that raw Step 1 scores U_beta, U_h and U_tau are present: report NA unless
// ScoreInfoAP() was executed in this process.
void GHAModel::WriteTauIFAudit(const string& prefix) const{
  if(!Asc || !psiinv_in || Asc->GetNData()!=psiinv_in->GetNBlocks())
    throw std::runtime_error("Tau IF audit requires original-block Asc from Step 1 score/info");
  std::ofstream summary((prefix + "_step1_summary.tsv").c_str());
  std::ofstream blocks((prefix + "_step1_blocks.tsv").c_str());
  if(!summary || !blocks)
    throw std::runtime_error("Unable to create tau IF audit files for prefix: " + prefix);
  summary << std::setprecision(17);
  blocks << std::setprecision(17);
  const size_t nblocks = psiinv_in->GetNBlocks();
  vector<size_t> pseudo_counts(nblocks,0);
  for(size_t i=0;i<LM.size();++i){
    if(LM[i]>=nblocks) throw std::runtime_error("Invalid LM original-block mapping");
    ++pseudo_counts[LM[i]];
  }
  double sumA2=0.0, sumA=0.0, maxAbsA=0.0;
  size_t largestA=0, nonzeroA=0;
  vector<size_t> ranked(nblocks,0);
  for(size_t i=0;i<nblocks;++i){
    ranked[i]=i;
    const double a=Asc->GetVal(i);
    if(!std::isfinite(a)) throw std::runtime_error("Non-finite Asc in tau IF audit");
    sumA+=a;
    sumA2+=a*a;
    if(a!=0.0) ++nonzeroA;
    if(std::fabs(a)>maxAbsA){maxAbsA=std::fabs(a);largestA=i;}
  }
  std::sort(ranked.begin(),ranked.end(),[this](size_t a,size_t b){
    return std::fabs(Asc->GetVal(a))>std::fabs(Asc->GetVal(b));
  });
  const size_t N=(Vinv_unq && Vinv_unq->GetNRows()>0) ?
      Vinv_unq->GetNRows()-1 : 0;
  const double tau_inv_diag=(Vinv_unq && Vinv_unq->GetNRows()>0) ?
      Vinv_unq->GetVal(N,N) : std::numeric_limits<double>::quiet_NaN();
  // For an exact inverse of the Step 1 information, this reciprocal is the
  // Schur complement after profiling out beta and baseline hazards.
  const bool schur_ok=!fixed_tau_mode && std::isfinite(tau_inv_diag) &&
      tau_inv_diag!=0.0;
  const double tau_schur=schur_ok ? 1.0/tau_inv_diag :
      std::numeric_limits<double>::quiet_NaN();
  double beta_l2=0, beta_max=0, h_l2=0, h_max=0, h_times_score_max=0;
  if(ap_scores_fresh){
    for(size_t i=0;i<Ubeta->GetNData();++i){
      const double x=Ubeta->GetVal(i);
      beta_l2+=x*x; beta_max=std::max(beta_max,std::fabs(x));
    }
    beta_l2=std::sqrt(beta_l2);
    if(Uh_unq && h_unq && Uh_unq->GetNData()==h_unq->GetNData())
      for(size_t i=0;i<Uh_unq->GetNData();++i){
        const double x=Uh_unq->GetVal(i);
        h_l2+=x*x; h_max=std::max(h_max,std::fabs(x));
        h_times_score_max=std::max(h_times_score_max,
             std::fabs(x*h_unq->GetVal(i)));
      }
    h_l2=std::sqrt(h_l2);
  }
  summary << "tau\tn_original_blocks\tn_pseudo_blocks\tn_nonzero_Asc"
          << "\tsum_Asc\tsum_Asc2\tsqrt_sum_Asc2\tmax_abs_Asc\tmax_abs_Asc_block_id"
          << "\tAP_schur_available\tAP_inv_tau_tau\tAP_schur_tau"
          << "\tAP_scores_fresh\tU_tau\tU_beta_l2\tU_beta_maxabs"
          << "\tU_h_l2\tU_h_maxabs\tmax_abs_h_times_Uh\n";
  summary << tau << '\t' << nblocks << '\t' << LM.size() << '\t' << nonzeroA
          << '\t' << sumA << '\t' << sumA2 << '\t' << std::sqrt(sumA2)
          << '\t' << maxAbsA << '\t' << largestA << '\t'
          << (schur_ok?"1":"0") << '\t';
  if(schur_ok) summary << tau_inv_diag << '\t' << tau_schur;
  else summary << "NA\tNA";
  summary << '\t' << (ap_scores_fresh?"1":"0") << '\t';
  if(ap_scores_fresh)
    summary << utau << '\t' << beta_l2 << '\t' << beta_max << '\t'
            << h_l2 << '\t' << h_max << '\t' << h_times_score_max;
  else summary << "NA\tNA\tNA\tNA\tNA\tNA";
  summary << '\n';
  blocks << "original_block_id\tblock_offset\tblock_size\tevents"
         << "\tpseudo_blocks_from_original\tAsc\tAsc_squared"
         << "\tAsc_squared_fraction\tabs_Asc_rank\n";
  vector<size_t> rank(nblocks,0);
  for(size_t j=0;j<ranked.size();++j) rank[ranked[j]]=j+1;
  for(size_t i=0;i<nblocks;++i){
    const size_t offset=psiinv_in->GetBlockOffset(i);
    const size_t n=psiinv_in->GetBlockRows(i);
    size_t events=0;
    for(size_t j=0;j<n;++j) if(delta_in->GetVal(offset+j)>0.0) ++events;
    const double a=Asc->GetVal(i);
    blocks << i << '\t' << offset << '\t' << n << '\t' << events
           << '\t' << pseudo_counts[i] << '\t' << a << '\t' << a*a
           << '\t' << (sumA2>0.0?a*a/sumA2:0.0)
           << '\t' << rank[i] << '\n';
  }
  std::cout << "[TAU_IF_AUDIT] tau=" << Log4(tau)
            << " sum_Asc2=" << Log4(sumA2) << " sqrt_sum_Asc2=" << Log4(std::sqrt(sumA2))
            << " top_block=" << largestA << " max_abs_Asc=" << Log4(maxAbsA)
            << " schur_tau=";
  if(schur_ok) std::cout << Log4(tau_schur); else std::cout << "NA";
  std::cout << " Step1_scores_fresh=" << (ap_scores_fresh?"yes":"no")
            << "\n[TAU_IF_AUDIT] files: " << prefix << "_step1_summary.tsv, "
            << prefix << "_step1_blocks.tsv" << std::endl;
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
  // pick the variance ratio bin for this variant
  vr_bin = GetVarRatioBin(MAF);
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
    
  g_projection_ready = false;
  g_tilde_ready      = false;
  g_working_var     = numeric_limits<double>::quiet_NaN();
  auto t1 = high_resolution_clock::now();
  titer_ist_us["SetG"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModel::ComputeScoreInfo(){
  auto t0 = high_resolution_clock::now();
  cout << "Computing AP score and information..." << endl;
  if(fixed_tau_mode){
    // Fixed-tau experiment: Step 1 is NOT estimated. Initialize the
    // shared event-time grid and AP-shaped placeholders needed by the
    // original Step 2 / influence-code paths. No AP score fitting.
    UpdateCLambda("UpdateAP_clambda", T_perm_t, lambda, clambda, T_sort);
    auto uniq = UpdateUnique("UpdateUniqueAP", index_unq,
                             delta_sort, T_sort, T_unq, lambda, h_unq);
    T_unq = uniq.first;
    h_unq = uniq.second;
    const size_t n_nuis = T_unq->GetNData() + beta->GetNData();
    if(V_unq) delete V_unq;
    if(Vinv_unq) delete Vinv_unq;
    V_unq = new PMatrix<double>("V_unq_known_tau", n_nuis+1,n_nuis+1);
    Vinv_unq = new PMatrix<double>("Vinv_unq_known_tau",n_nuis+1,n_nuis+1);
    V_unq->Zero();
    Vinv_unq->Zero();
    Asc->Zero();
    utau = 0.0;
    Ubeta->Zero();
    if(Uh_unq) delete Uh_unq;
    Uh_unq = new PMatrix<double>("Uh_unq_known_tau",T_unq->GetNData(),1);
    Uh_unq->Zero();
    cout << "[EXPERIMENT_FIXED_TAU] Asc=0, known tau=" << tau
         << "; initialized shared matrices without AP influence"
         << "; NOT for production inference" << endl;
  }else{
    ScoreInfoAP();
  }
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
  niterations ++;
  ugamma      = 0.0;
  ugamma_cent = 0.0;
  ugamma_var  = 0.0;
  vr_audit_record = VRAuditRecord();
  const size_t L = beta->GetNData();
  const size_t M = T_unq->GetNData();
  const size_t N = T_unq->GetNData() + beta->GetNData();
  // Keep the original association score definition on centered G_in.
  // Build G_tilde only for the cheap VR working variance and SPA CGF.
  // This deliberately separates the exact estimating equation from the
  // working approximation used for VR/SPA.
  BuildResidualizedG();
  if(!g_tilde_ready)
    throw std::runtime_error("Residualized working genotype was not initialized");

  buffvec_in->Zero();
  for(size_t i=0; i<delta_in->GetNData(); ++i)
    buffvec_in->SetVal(i, delta_in->GetVal(i) -
      clambda_in->GetVal(i)*ebxw1_in->GetVal(i));

  vector<double> utG;
  StridedInnerProduct(buffvec_in, G_in, psiinv_in->GetBlockRows(),
                      utG, 0.0, 0.0, bnorm_ugamma);
  ugamma = accumulate(utG.begin(), utG.end(), 0.0);

  // Diagnostic only: the SPA/VR working score should be numerically close
  // to the exact centered-G score when the nuisance equations are solved.
  if(vr_diagnostics){
    vector<double> residualized_block_score;
    StridedInnerProduct(buffvec_in, G_tilde_in, psiinv_in->GetBlockRows(),
                        residualized_block_score, 0.0, 0.0, bnorm_ugamma);
    ugamma_cent = accumulate(residualized_block_score.begin(),
                             residualized_block_score.end(), 0.0);
    cout << "[SCORE_SCALE] MAF=" << Log4(MAF)
         << " U_exact_centered=" << Log4(ugamma)
         << " U_working_residualized=" << Log4(ugamma_cent)
         << " delta=" << Log4(ugamma_cent - ugamma) << endl;
  }
  // decide whether to perform the full calculation
  bool full_calc = true;
  if(!vr_audit_enabled && use_var_ratio && vr_bin < var_ratio_found.size() &&
     var_ratio_found[vr_bin]){
    full_calc = false;
    ugamma_var = ComputeVarRatio();
    if(ugamma_var == 0.0 && (ugamma != 0.0 || ugamma_cent != 0.0)){
      full_calc = true;
      ugamma_var = 0.0;
    }
  }
  // Prepare FD lazily if a tagged precomputed VR required no up-front FD,
  // but this specific variant needs exact block-sandwich variance.
  if(full_calc && tau_fd_cross_enabled && !tau_fd_ready){
    std::cout << "[GADO_TAU_CROSS] Exact-variance fallback: computing derivative once." << std::endl;
    PrepareTauFD("");
    // PrepareTauFD intentionally invalidates the working-weight cache.
    // Rebuild it for the current SNP before computing the exact variance/SPA.
    BuildResidualizedG();
    if(!g_tilde_ready)
      throw std::runtime_error("Residualized working genotype unavailable after tau FD preparation");
  }
  // perform the full variance calculation if necessary
  if(full_calc){
    double audit_raw = 0.0;
    double audit_cross = 0.0;
    double audit_correction = 0.0;
    double audit_no_tau_corr = 0.0;
    double audit_sum_asc2_scaled = 0.0;
    double audit_sum_v_asc_scaled = 0.0;
    double audit_max_tau_squared_block = 0.0;
    double audit_max_block_contribution = 0.0;
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
    I_G->Zero();
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
	  if(!AllVarRatioFound() && cache_SI)
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
      // Experimental cross-information pathway: replace only the last row
      // of SNP-to-nuisance information with the derivative of the IMPLEMENTED
      // approximate score at fixed beta/Lambda. Do not change the nuisance
      // information H or the original stored V_unq_in matrix.
      if(hq_fd_audit_enabled){
        hq_analytic_q.assign(N,0.0);
        for(size_t t=0;t<N;++t) hq_analytic_q[t]=I_G->GetVal(t);
      }
      const double audit_I_gamma_tau_analytic_raw = I_G->GetVal(N);
      if(tau_fd_cross_enabled){
        if(!tau_fd_ready || tau_fd_direct_coef.size()!=nind || tau_fd_c.size()!=N)
          throw std::runtime_error("Experimental tau FD cross requires PrepareTauFD and matching dimensions");
        double direct_fd = 0.0;
        for(size_t ii=0; ii<nind; ++ii)
          direct_fd += G_in->GetVal(ii)*tau_fd_direct_coef[ii];
        if(!std::isfinite(direct_fd))
          throw std::runtime_error("Experimental tau FD direct information is nonfinite");
        I_G->SetVal(N,direct_fd);
      }
      // Diagnostic-only projection of the *entire* tau-estimation influence
      // through the nuisance block. Simply dropping Stheta[N]*I_G[N]
      // is wrong: tau also contributes to the first N entries via
      // -Asc_b * Vinv_unq_in * V_unq_in[0:N,N].
      double audit_tau_adjust_coeff = 0.0;
      double audit_tau_projection_analytic = 0.0;
      double audit_tau_projection_fd_c = std::numeric_limits<double>::quiet_NaN();
      double audit_I_gamma_tau_fd = std::numeric_limits<double>::quiet_NaN();
#ifndef __CUDA__
      if(vr_audit_enabled){
        audit_tau_adjust_coeff = I_G->GetVal(N);
        for(size_t row=0;row<N;++row){
          double beta_lambda_projection=0.0;
          for(size_t col=0;col<N;++col)
            beta_lambda_projection += Vinv_unq_in->GetVal(row,col) *
                (tau_fd_cross_enabled ? tau_fd_c[col] :
                                          V_unq_in->GetVal(col,N));
          audit_tau_adjust_coeff -= beta_lambda_projection * I_G->GetVal(row);
          audit_tau_projection_analytic += beta_lambda_projection * I_G->GetVal(row);
        }
        vr_audit_record.I_gamma_tau_analytic=audit_I_gamma_tau_analytic_raw;
        // Preserve the historic 'analytic projection' field as analytic,
        // even though the experimental correction uses FD c.
        if(tau_fd_cross_enabled){
          audit_tau_projection_analytic=0.0;
          for(size_t row=0; row<N; ++row){
            double orig_proj=0.0;
            for(size_t col=0; col<N; ++col)
              orig_proj += Vinv_unq_in->GetVal(row,col) *
                           V_unq_in->GetVal(col,N);
            audit_tau_projection_analytic += orig_proj*I_G->GetVal(row);
          }
        }
        vr_audit_record.tau_projection_analytic=audit_tau_projection_analytic;
        if(tau_fd_ready){
          if(tau_fd_direct_coef.size()!=nind || tau_fd_c.size()!=N)
            throw std::runtime_error("Tau FD dimensions do not match SNP audit");
          audit_I_gamma_tau_fd=0.0;
          for(size_t i=0;i<nind;++i)
            audit_I_gamma_tau_fd += G_in->GetVal(i)*tau_fd_direct_coef[i];
          audit_tau_projection_fd_c=0.0;
          for(size_t row=0;row<N;++row){
            double bpc=0.0;
            for(size_t col=0;col<N;++col)
              bpc+=Vinv_unq_in->GetVal(row,col)*tau_fd_c[col];
            audit_tau_projection_fd_c+=I_G->GetVal(row)*bpc;
          }
          vr_audit_record.I_gamma_tau_fd=audit_I_gamma_tau_fd;
          vr_audit_record.tau_projection_fd_c=audit_tau_projection_fd_c;
          vr_audit_record.h_tau_fd_c=audit_I_gamma_tau_fd-audit_tau_projection_fd_c;
          vr_audit_record.I_gamma_tau_gap=
            audit_I_gamma_tau_fd-audit_I_gamma_tau_analytic_raw;
          vr_audit_record.h_tau_fd_c_gap=
            vr_audit_record.h_tau_fd_c-audit_tau_adjust_coeff;
          if(tau_fd_cross_enabled &&
             std::fabs(vr_audit_record.h_tau_fd_c-audit_tau_adjust_coeff) >
               1.e-7*std::max(1.,std::fabs(audit_tau_adjust_coeff)))
            throw std::runtime_error("Experimental tau FD h mismatch with independent FD audit");
        }
      }
#endif
      // compute S_theta
      for(size_t i=0; i<psiinv_in->GetNBlocks(); i++){
	if(use_cache) Stheta = Stheta_cache[i];
	else{
	  V_unq_in->GetBlock(0, N, Stmp);
        if(tau_fd_cross_enabled){
          // FD nuisance-to-tau cross-information, with H^{-1} unchanged.
          // The original matrix V_unq_in remains unchanged for diagnostics
          // and score-info output. Stheta_cache stores this corrected vector.
          for(size_t z=0; z<N; ++z)
            Stmp->SetVal(z,tau_fd_c[z]);
        }
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
	  if(!AllVarRatioFound() && cache_SI)
	    Stheta_cache.push_back(new PMatrix<double>("Stheta_"+to_string(i),
						       *Stheta));
#ifdef __CUDA__
	  for(size_t l=0; l<Stheta->GetNData(); l++)
	    gStheta->SetVal(l, i, Stheta->GetVal(l));
#endif	
	}
	const double correction_i = Stheta->InnerColProduct(0, 0, *I_G);
	tmp2 = utG[i] - correction_i;
#ifndef __CUDA__
        if(vr_audit_enabled){
          const double scale = bnorm_ugamma ?
            sqrt(static_cast<double>(psiinv_in->GetBlockRows(i))) : 1.0;
          audit_raw += utG[i]*utG[i]/scale;
          audit_cross += utG[i]*correction_i/scale;
          audit_correction += correction_i*correction_i/scale;
          const double tau_correction_i =
            Asc->GetVal(i) * audit_tau_adjust_coeff;
          const double no_tau = tmp2 + tau_correction_i;
          audit_no_tau_corr += no_tau*no_tau/scale;
          const double asc = Asc->GetVal(i);
          audit_sum_asc2_scaled += asc*asc/scale;
          audit_sum_v_asc_scaled += no_tau*asc/scale;
          const double tau_block_squared =
              tau_correction_i*tau_correction_i/scale;
          if(tau_block_squared > audit_max_tau_squared_block){
            audit_max_tau_squared_block = tau_block_squared;
            vr_audit_record.top_tau_block_id = static_cast<long long>(i);
            vr_audit_record.top_tau_block_size = psiinv_in->GetBlockRows(i);
            const size_t base = psiinv_in->GetBlockOffset(i);
            size_t n_events = 0;
            for(size_t s=0; s<psiinv_in->GetBlockRows(i); ++s)
              if(delta_in->GetVal(base+s)>0.0) ++n_events;
            vr_audit_record.top_tau_block_events = n_events;
          }
          const double block_var = tmp2*tmp2/scale;
          if(block_var > audit_max_block_contribution){
            audit_max_block_contribution = block_var;
            vr_audit_record.dominant_block_id = static_cast<long long>(i);
            const size_t bsize = psiinv_in->GetBlockRows(i);
            const size_t boffset = psiinv_in->GetBlockOffset(i);
            size_t events = 0;
            for(size_t k=0; k<bsize; ++k)
              if(delta_in->GetVal(boffset+k)>0.0) ++events;
            vr_audit_record.dominant_block_size = bsize;
            vr_audit_record.dominant_block_events = events;
          }
        }
#endif
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
#ifndef __CUDA__
    if(vr_audit_enabled){
      vr_audit_record.valid = true;
      vr_audit_record.raw = audit_raw;
      vr_audit_record.cross = audit_cross;
      vr_audit_record.correction = audit_correction;
      vr_audit_record.exact = ugamma_var;
      vr_audit_record.no_tau_correction = audit_no_tau_corr;
      vr_audit_record.tau_adjust_coeff = audit_tau_adjust_coeff;
      vr_audit_record.sum_asc2_scaled = audit_sum_asc2_scaled;
      vr_audit_record.sum_v_asc_scaled = audit_sum_v_asc_scaled;
      vr_audit_record.tau_cross_term =
          -2.0*audit_tau_adjust_coeff*audit_sum_v_asc_scaled;
      vr_audit_record.tau_squared_term =
          audit_tau_adjust_coeff*audit_tau_adjust_coeff*audit_sum_asc2_scaled;
      vr_audit_record.tau_decomposition_error = std::fabs(
          ugamma_var - (audit_no_tau_corr +
              vr_audit_record.tau_cross_term +
              vr_audit_record.tau_squared_term));
      vr_audit_record.top_tau_block_fraction =
          vr_audit_record.tau_squared_term>0.0 ?
          audit_max_tau_squared_block/vr_audit_record.tau_squared_term : 0.0;
      vr_audit_record.working = g_working_var;
      vr_audit_record.ratio = g_working_var>0.0 ?
         ugamma_var/g_working_var : 0.0;
      vr_audit_record.algebra_error = std::fabs(
          ugamma_var - (audit_raw - 2.0*audit_cross + audit_correction));
      vr_audit_record.dominant_block_fraction = ugamma_var>0.0 ?
         audit_max_block_contribution/ugamma_var : 0.0;
      // Weighted score-orthogonality residual scaled to [0,1] in exact
      // arithmetic (Cauchy-Schwarz), for [intercept, X] nuisance columns.
      const size_t q = X_in->GetNCols()+1;
      for(size_t j=0; j<q; ++j){
        double weighted_dot=0.0, cov_norm2=0.0;
        for(size_t k=0; k<nind; ++k){
          const double x = (j==0 ? 1.0 : X_in->GetVal(k,j-1));
          const double w = ub_in->GetVal(k);
          weighted_dot += w*x*G_tilde_in->GetVal(k);
          cov_norm2 += w*x*x;
        }
        if(cov_norm2>0.0 && g_working_var>0.0)
          vr_audit_record.orthogonality_max = std::max(
            vr_audit_record.orthogonality_max,
            std::fabs(weighted_dot)/std::sqrt(cov_norm2*g_working_var));
      }
      // PSD heuristic only: C=(1-rho)I+rho*GRM, rho=tau/(1+tau).
      // This is NOT the derived frailty score covariance; for diagnostics
      // and held-out comparison ONLY, never for a production p-value.
      const double rho = tau/(1.0+tau);
      double quadratic=0.0;
      PMatrix<double> grm_block;
      for(size_t bi=0; bi<psi_in->GetNBlocks(); ++bi){
        psi_in->GetBlock(bi,grm_block);
        const size_t n = psi_in->GetBlockRows(bi);
        const size_t offset = psi_in->GetBlockOffset(bi);
        for(size_t a=0;a<n;++a){
          const size_t ia=offset+a;
          const double ya=G_tilde_in->GetVal(ia)*std::sqrt(ub_in->GetVal(ia));
          for(size_t b=0;b<n;++b){
            const size_t ib=offset+b;
            const double yb=G_tilde_in->GetVal(ib)*std::sqrt(ub_in->GetVal(ib));
            quadratic += ya*grm_block.GetVal(a,b)*yb;
          }
        }
      }
      vr_audit_record.block_proxy=(1.0-rho)*g_working_var+rho*quadratic;
      vr_audit_record.ratio_block_proxy = vr_audit_record.block_proxy>0.0 ?
          ugamma_var/vr_audit_record.block_proxy : 0.0;
      if(vr_audit_record.algebra_error >
         1e-8*std::max(1.0,std::fabs(ugamma_var)))
        throw std::runtime_error("VR audit decomposition identity failed");
      if(vr_audit_record.tau_decomposition_error >
         1e-8*std::max(1.0,std::fabs(ugamma_var)))
        throw std::runtime_error("Tau IF variance decomposition identity failed");
    }
#else
    if(vr_audit_enabled)
      throw std::runtime_error("--vr-audit-tsv is CPU-only");
#endif
    if(use_var_ratio) ComputeVarRatio();
  }
  // Score statistic uses the original centered-G exact score.
  // SPA keeps its G_tilde working CGF; under the fitted null, the two scores
  // differ only by nuisance-score directions and should be numerically close.
  if(!std::isfinite(ugamma) || !std::isfinite(ugamma_var) ||
     ugamma_var < 0.0)
    throw std::runtime_error("Non-finite or negative score variance");
  if(ugamma_var == 0.0){
    if(ugamma != 0.0)
      throw std::runtime_error("Nonzero score with zero exact/VR variance");
    score_stat = 0.0;
  }else{
    score_stat = ugamma / std::sqrt(ugamma_var);
  }

  // The SPA CGF is built from residualized G and working weights W.
  // Hence K''(0) = D = G_tilde' W G_tilde, while the final score
  // variance is V = r * D. This scaling must also hold when the
  // exact variance is used for a calibration SNP / exact fallback.
  bool spa_supported = true;
  double vr_use = 1.0;
  if(!full_calc && use_var_ratio && vr_bin < var_ratio_found.size() &&
     var_ratio_found[vr_bin]){
    vr_use = vr_mean[vr_bin];
  }else if(ugamma_var > 0.0){
    const double working_var = g_working_var;
    if(std::isfinite(working_var) && working_var > 0.0)
      vr_use = ugamma_var / working_var;
    else
      spa_supported = false; // Cannot match CGF scale to exact variance.
  }
  spa_supported = spa_supported && std::isfinite(vr_use) && vr_use > 0.0;

  if(vr_diagnostics && print){
    cout << "[SCORE_VARIANCE_SCALE] exact_path=" << full_calc
         << " variance=" << Log4(ugamma_var) << " spa_ratio=" << Log4(vr_use)
         << " spa_supported=" << spa_supported << endl;
  }

  if(ugamma_var == 0.0 || abs(score_stat) < spa_thresh || !spa_supported){
    pval = (ugamma_var == 0.0) ? 1.0 : TMath::Prob(score_stat*score_stat, 1);
  }else{
    pval = ComputeWeightedP(ugamma, ugamma_var, vr_use);
  }

  // score stat and p-value for centralized case
//   if(ugamma_cent == 0.0 && ugamma_var == 0.0) score_stat_cent = 0.0;
//   else score_stat_cent = ugamma_cent / sqrt(ugamma_var);
//   // if((full_calc && (use_var_ratio ||
// 		    (abs(score_stat_cent) <= spa_thresh))) ||
//      (!full_calc && (abs(score_stat_cent) <= spa_thresh)))
//     pval_cent = TMath::Prob(score_stat_cent*score_stat_cent, 1);
//   else
//     pval_cent = ComputePValue(ugamma_cent, ugamma_var, vr_use);
  // Keep the original internal timer; routine logs show only one compact
  // SNP result line (the driver already reports the variant index).
  auto t1 = high_resolution_clock::now();
  size_t ns = duration_cast<nanoseconds>(t1-t0).count();
  ns_st += ns;
  if(print){
    cout << "MAC=" << Log4(MAC)
         << "; MAF=" << Log4(MAF)
         << "; p-value=" << Log4(pval)
         << "; score=" << Log4(score_stat)
         << "; Ugamma=" << Log4(ugamma)
         << "; Ugamma_var=" << Log4(ugamma_var)
         << endl;
  }
  titer_ist_us["ComputeScoreTest"] += ns * 1e-3;
}


double GHAModel::ComputeVarRatio(){
  auto t0 = high_resolution_clock::now();
  const size_t bin = vr_bin;

  // The centered genotype is used in the exact score and block variance.
  // The same residualized G_tilde is used for this cheap working variance
  // and the SPA CGF, preserving K''(0) = D.
  BuildResidualizedG();
  if(!g_tilde_ready)
    throw std::runtime_error("Residualized genotype not available for variance ratio");

  // Already computed directly and cached during residualization.
  // Do not perform another O(n) pass for each genome-wide variant.
  const double GuG = g_working_var;
  if(!std::isfinite(GuG) || GuG <= 1e-8){
    if(vr_diagnostics)
      cerr << "VR working variance invalid: global calibration"
           << " MAF=" << Log4(MAF) << " D=" << Log4(GuG) << endl;
    return 0.0;
  }

  // ============================================================
  // 6. Calibrate the one global variance ratio.
  //    Only accepted, finite, positive marker-specific ratios count.
  //    We MUST NOT recheck an unchanged median after an invalid marker.
  // ============================================================

  // Exclude out-of-range MAFs from the calibration sample. Variants
  // outside the specified calibration range may still be tested using
  // whichever bin the upstream workflow assigns to them.
  const bool calibration_maf_ok =
    std::isfinite(MAF) &&
    !vr_maf_edges.empty() &&
    MAF >= vr_maf_edges.front() && MAF <= vr_maf_edges.back();

  if(use_var_ratio &&
     bin < var_ratio_found.size() &&
     !var_ratio_found[bin] &&
     calibration_maf_ok &&
     std::isfinite(ugamma_var) && ugamma_var > 0.0){

    const double r = ugamma_var / GuG;
    if(std::isfinite(r) && r > 0.0){
      var_ratio[bin].push_back(r);
      const size_t n = var_ratio[bin].size();

      if(vr_diagnostics){
        cout << "VR marker: bin=" << bin
             << " accepted=" << n
             << " MAF=" << Log4(MAF) << " MAC=" << Log4(MAC)
             << " exact=" << Log4(ugamma_var)
             << " working=" << Log4(GuG)
             << " ratio=" << Log4(r) << endl;
      }

      const size_t initial = std::max<size_t>(1, vr_init_eval);
      const size_t step = std::max<size_t>(1, vr_step_eval);
      const size_t maximum = std::max(initial, vr_max_eval);
      const size_t required = std::max<size_t>(1, vr_stable_checks);
      const bool evaluate_ratio =
        (n == initial) ||
        (n > initial && ((n - initial) % step == 0));

      if(evaluate_ratio){
        vector<double> sorted = var_ratio[bin];
        sort(sorted.begin(), sorted.end());
        const size_t m = sorted.size();
        const double med = (m % 2 == 0)
          ? 0.5 * (sorted[m/2-1] + sorted[m/2]) : sorted[m/2];
        const double q25 = sorted[m/4];
        const double q75 = sorted[(3*m)/4];

        double rel_change = numeric_limits<double>::quiet_NaN();
        if(std::isfinite(vr_prev[bin])){
          rel_change = std::abs(med - vr_prev[bin]) /
            std::max(1e-12, std::abs(vr_prev[bin]));
        }

        // First median at n=initial is the reference; after that,
        // require two successive stable updates. No fixed minimum
        // marker count beyond the initial checkpoint is imposed.
        if(std::isfinite(rel_change) && rel_change < vr_epsilon)
          ++vr_stable_count[bin];
        else
          vr_stable_count[bin] = 0;

        cout << "Global VR calibration [" << Log4(vr_maf_edges.front())
             << "," << Log4(vr_maf_edges.back()) << "]"
             << " median=" << Log4(med)
             << " IQR=[" << Log4(q25) << ", " << Log4(q75) << "]"
             << " rel_change=" << Log4(rel_change)
             << " stable=" << vr_stable_count[bin]
             << "/" << required << endl;

        vr_prev[bin] = med;

        if(vr_stable_count[bin] >= required){
          // Do not silently truncate large estimates (the original
          // std::min(med, 10.0) hid potentially unstable calibrations).
          vr_mean[bin] = med;
          var_ratio_found[bin] = true;
          if(med > 10.0)
            cerr << "WARNING: large converged global variance ratio: rhat=" << Log4(med)
                 << ". Check numerator/denominator and marker QC." << endl;
          cout << "Global VR converged: rhat=" << Log4(med) << endl;

          // Release exact-variance caches once the global ratio converges.
          if(AllVarRatioFound()){
            for(auto s : Stheta_cache) if(s) delete s;
            for(auto I : Igamma_cache) if(I) delete I;
            Stheta_cache.clear();
            Igamma_cache.clear();
          }
        }else if(n >= maximum){
          // Fail closed rather than using an unreliable bin-specific ratio.
          ostringstream msg;
          msg << "Variance-ratio calibration did not converge for MAF bin "
              << bin << " after " << n << " valid markers"
              << " (median=" << med << ", stable_checks="
              << vr_stable_count[bin] << "/" << required
              << "). Inspect QC, denominator, and calibration pool. "
              << "Adjust VR limits only after inspecting diagnostics.";
          throw runtime_error(msg.str());
        }
      }
    }else if(vr_diagnostics){
      cerr << "VR marker rejected: bin=" << bin
           << " MAF=" << Log4(MAF) << " MAC=" << Log4(MAC)
           << " exact=" << Log4(ugamma_var)
           << " working=" << Log4(GuG) << " ratio=" << Log4(r) << endl;
    }
  }


  auto t1 = high_resolution_clock::now();

  titer_ist_us["ComputeVarRatio"] =
    duration_cast<nanoseconds>(t1-t0).count() * 1.0e-3;


  // ============================================================
  // genome-wide approximate variance for this bin
  //
  // V = rhat[bin] * Gtilde' W Gtilde
  // ============================================================

  return (bin < vr_mean.size() ? vr_mean[bin] : 0.0) * GuG;
}


double GHAModel::ComputeWeightedP(double ug, double var, double vr)
{
  double z = ug / std::sqrt(var);
  double absz = std::fabs(z);
  
  double p_normal = TMath::Prob(z*z, 1);
  double p_spa    = ComputePValue(ug, var, vr);
  
  double c0 = 2.0;
  double delta = 0.5;
  
  double w = (absz - (c0 - delta)) / (2.0 * delta);
  if(w < 0.0) w = 0.0;
  if(w > 1.0) w = 1.0;
  
  return (1.0 - w) * p_normal + w * p_spa;
}

double GHAModel::ComputePValue(
    const double& ug,
    const double& var,
    const double& var_ratio
){
  auto t0 = high_resolution_clock::now();

  // =========================================================
  // Normal two-sided fallback
  //
  // var should be the variance-ratio-adjusted score variance:
  // var = var_ratio * variance_without_GRM
  // =========================================================

  auto two_side_fallback = [&]() {

    if(
      !std::isfinite(ug) ||
      !std::isfinite(var) ||
      var <= 0.0
    ){
      return 1.0;
    }

    double p = TMath::Prob(
      (ug * ug) / var,
      1
    );

    if(!std::isfinite(p)){
      return 1.0;
    }

    return std::min(
      1.0,
      std::max(
        0.0,
        p
      )
    );
  };


  // =========================================================
  // Input validation
  // =========================================================

  if(
    !std::isfinite(ug) ||
    !std::isfinite(var) ||
    var <= 0.0 ||
    !std::isfinite(var_ratio) ||
    var_ratio <= 0.0
  ){
    return two_side_fallback();
  }


  // =========================================================
  // Critical correction:
  //
  // Official GATE correction:
  // SPA score target S is divided by sqrt(varRatio).
  //
  // Keep ComputeCGFs() on its existing raw-score scale,
  // and rescale only the target supplied to the saddlepoint
  // equation.
  // =========================================================

  const double spa_abs_score =
    std::abs(ug) /
    std::sqrt(var_ratio);


  auto one_side_fallback = [&]() {

    // Symmetric split of the adjusted normal two-sided p-value
    return 0.5 * two_side_fallback();
  };


  double p_two = 0.0;


  // =========================================================
  // Compute lower and upper tails
  // =========================================================

  for(int i = -1; i <= 1; i += 2){

    // FIXED:
    // old: s = ug * i
    // new: s = |ug| / sqrt(var_ratio) * i
    double s = spa_abs_score * static_cast<double>(i);

    double k0 = 0.0;
    double k1 = 0.0;
    double k1p = 0.0;
    double k2 = 0.0;

    double c1 = 0.0;
    double c1p =
      std::numeric_limits<double>::max();

    double alpha = 0.0;
    double alphap =
      std::numeric_limits<double>::max();

    size_t nrit = 0;

    // Initial CGF values at c1 = 0
    ComputeCGFs(
      c1,
      k0,
      k1,
      k2
    );


    // =======================================================
    // Solve K'(c1) = s
    // =======================================================

    while(
      std::abs(c1 - c1p) >= 1e-4
    ){

      if(
        nrit++ >= 1000 ||
        !std::isfinite(c1) ||
        !std::isfinite(k1) ||
        !std::isfinite(k2) ||
        k2 <= 0.0
      ){
        return two_side_fallback();
      }


      alpha = (k1 - s) / k2;

      c1p = c1;
      c1 -= alpha;

      k1p = k1;

      ComputeCGFs(
        c1,
        k0,
        k1,
        k2
      );


      // Step-halving when the root is crossed
      if(
        std::signbit(k1 - s) !=
        std::signbit(k1p - s)
      ){

        if(
          std::abs(alpha) >= alphap
        ){

          alphap /= 2.0;

          c1 =
            c1p +
            (
              std::signbit(k1 - k1p)
                ? -1.0
                : 1.0
            ) * alphap;

          ComputeCGFs(
            c1,
            k0,
            k1,
            k2
          );

        }else{

          alphap = std::abs(alpha);
        }
      }
    }


    double p_side = 0.0;

    ComputeCGFs(
      c1,
      k0,
      k1,
      k2
    );


    // =======================================================
    // Validate CGF derivatives
    // =======================================================

    if(
      !std::isfinite(k1) ||
      !std::isfinite(k2) ||
      k2 <= 0.0
    ){

      p_side = one_side_fallback();

    }else{

      double temp =
        c1 * s - k0;


      // Protect the signed-root calculation
      if(
        !std::isfinite(temp) ||
        temp <= 0.0
      ){

        p_side = one_side_fallback();

      }else{

        double c2 =
          (
            std::signbit(c1)
              ? -1.0
              : 1.0
          ) *
          std::sqrt(
            2.0 * temp
          );


        double c3 =
          c1 *
          std::sqrt(k2);


        if(
          !std::isfinite(c2) ||
          !std::isfinite(c3) ||
          std::abs(c2) < 1e-10 ||
          (c3 / c2) <= 0.0
        ){

          p_side = one_side_fallback();

        }else{

          double Z =
            c2 +
            std::log(
              c3 / c2
            ) / c2;


          if(!std::isfinite(Z)){

            p_side = one_side_fallback();

          }else{

            // s > 0: upper tail
            // s < 0: lower tail
            bool lower_tail =
              (s < 0.0);


            p_side = R::pnorm(
              Z,
              0.0,
              1.0,
              lower_tail,
              false
            );


            if(!std::isfinite(p_side)){
              p_side = one_side_fallback();
            }
          }
        }
      }
    }


    p_two += p_side;
  }


  auto t1 =
    high_resolution_clock::now();

  titer_ist_us["ComputePValue"] =
    duration_cast<nanoseconds>(
      t1 - t0
    ).count() * 1.e-3;


  // =========================================================
  // Final two-sided guard
  // =========================================================

  if(
    !std::isfinite(p_two) ||
    p_two < 0.0 ||
    p_two > 1.0
  ){
    return two_side_fallback();
  }


  return p_two;
}

void GHAModel::ComputeCGFs(
    const double& c1,
    double& k0,
    double& k1,
    double& k2
){

  // SPA/VR working approximation uses cached G_tilde.
  // The exact score itself remains defined on centered G_in.
  BuildResidualizedG();
  if(!g_tilde_ready)
    throw std::runtime_error("SPA requested without residualized genotype");

  // ============================================================
  // CGF
  //
  // K(t)
  // =
  // sum_i u_i
  // [
  //   exp(t Gtilde_i)
  //   -
  //   t Gtilde_i
  //   -
  //   1
  // ]
  //
  // Therefore:
  //
  // K''(0)
  // =
  // Gtilde' W Gtilde
  //
  // exactly matching the denominator used by ComputeVarRatio().
  // ============================================================

  k0 = 0.0;
  k1 = 0.0;
  k2 = 0.0;


  const size_t n =
      clambda_in->GetNData();


  for(size_t i=0; i<n; i++){

    const double u =
        ub_in->GetVal(i);


    const double G =
        G_tilde_in->GetVal(i);


    const double c1G =
        c1 *
        G;


    const double ec1G =
        std::exp(c1G);


    k0 +=
        u *
        (
          ec1G -
          c1G -
          1.0
        );


    k1 +=
        u *
        G *
        (
          ec1G -
          1.0
        );


    k2 +=
        u *
        G *
        G *
        ec1G;
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
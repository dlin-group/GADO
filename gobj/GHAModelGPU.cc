#include "GHAModelGPU.hh"
#include "gha_kernels.h"
#include <Math/GSLRndmEngines.h>
#include <H5Cpp.h>
#include <assert.h>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>

using namespace std;
using namespace std::chrono;

GHAModelGPU::GHAModelGPU(string ghpath, string ghbase,
			 size_t ghdim, size_t ghorder,
			 GMatrix<double>* t, GMatrix<double>* d,
			 GMatrix<double>* x, BGMatrix<double>* p,
			 cublasHandle_t chandle){
  handle = chandle;
  for(size_t i=ghdim; i>0; i--){
    gauss_hermite.push_back(GaussHermiteND(ghpath, ghbase, ghorder, i));
    gh_roots_sqrt2.push_back(NULL);
    cudaMalloc((void**) &gh_roots_sqrt2.back(),
	       sizeof(double)*gauss_hermite.back().GetPowers()[0]);
    cudaMemcpy(gh_roots_sqrt2.back(),
	       &gauss_hermite.back().GetRootsSqrt2().front(),
	       sizeof(double)*gauss_hermite.back().GetPowers()[0],
	       cudaMemcpyHostToDevice);
    gh_powers.push_back(NULL);
    cudaMalloc((void**) &gh_powers.back(),
	       sizeof(size_t)*gauss_hermite.back().GetPowers().size());
    cudaMemcpy(gh_powers.back(),
	       &gauss_hermite.back().GetPowers().front(),
	       sizeof(size_t)*gauss_hermite.back().GetPowers().size(),
	       cudaMemcpyHostToDevice);
  }
  // setup clocking
  vector<string> parts({
      "Initialize", "SetSeedsAP", "TimeSortDataAP", "UpdateAP_aq",
      "UpdateAP_w1q", "UpdateAP_ebxw1q", "UpdateAP_beta", "UpdateAP_lambda",
      "UpdateAP_clambda", "UpdateAP_tau", "CheckConvergenceAP", "IterateAP",
      "SetSeedsIn", "TimeSortDataIn", "UpdateIn_b", "UpdateIn_mus",
      "UpdateIn_ebxw1", "UpdateIn_w1", "UpdateIn_beta", "UpdateIn_lambda",
      "UpdateIn_clambda", "IterateIn_b", "IterateIn", "CheckConvergenceIn"});
  for(auto const& p : parts) titer_us[p]  = 0.0;
  auto t0 = high_resolution_clock::now();
  // check dimensions
  assert(t->GetNRows() == d->GetNRows());
  assert(t->GetNCols() == 1 && d->GetNCols() == 1);
  assert(x->GetNRows() == t->GetNRows());
  assert(p->GetNRows() == x->GetNRows() && p->GetNCols() == x->GetNRows());
  // set data variables
  nind = t->GetNRows();
  npar = x->GetNCols();
  b_last   = true;
  b_fullnr = true;
  verbose  = true;
  T_in     = t;
  delta_in = d;
  X_in     = x;
  psi_in   = p;
  // expand psi into the pair structure
  psi = new BGMatrix<double>("psi_ap", handle);
  PMatrix<double>* block  = new PMatrix<double>();
  PMatrix<double>* sblock = new PMatrix<double>("sblock", 2, 2, false);
  size_t offset = 0;
  vector<size_t> index_perm = psi_in->GetIndexPermutation();
  for(size_t i=0; i<psi_in->GetNBlocks(); i++){
    psi_in->GetBlock(i, *block);
    if(block->GetNRows() <= gauss_hermite.size()){
      psi->AppendBlock(block);
      size_t tmp = index_map.size();
      index_map.resize(tmp + block->GetNRows());
      iota(index_map.begin()+tmp, index_map.end(), offset); 
    }
    else{
      for(size_t j=0; j<block->GetNRows(); j++)
	for(size_t k=j+1; k<block->GetNRows(); k++){
	  sblock->SetVal(0, 0, block->GetVal(j, j));
	  sblock->SetVal(0, 1, block->GetVal(j, k));
	  sblock->SetVal(1, 0, block->GetVal(k, j));
	  sblock->SetVal(1, 1, block->GetVal(k, k));
	  psi->AppendBlock(sblock);
	  index_map.push_back(index_perm[offset+j]);
	  index_map.push_back(index_perm[offset+k]);
	}
    }
    offset += block->GetNRows();
  }
  if(block)  delete block;
  if(sblock) delete sblock;
  psi->CloseToAppend();
  psi->ChangeDevLock(false);
  psi_in->ChangeDevLock(false);
  // setup expanded input data
  T     = new GMatrix<double>("T",     psi->GetNRows(), 1, false, handle);
  delta = new GMatrix<double>("delta", psi->GetNRows(), 1, false, handle);
  X     = new GMatrix<double>("X",     psi->GetNRows(),
			      X_in->GetNCols(),false, handle);
  for(size_t i=0; i<index_map.size(); i++){
    size_t j = index_map[i];
    T->SetVal(i, T_in->GetVal(j));
    delta->SetVal(i, delta_in->GetVal(j));
    for(size_t k=0; k<X->GetNRows(); k++)
      X->SetVal(i, k, X_in->GetVal(j, k));
  }
  // sort the blocks by size, reorder index map, and permute expanded data
  vector<size_t> index_tmp(index_map.begin(), index_map.end());
  fill(index_map.begin(), index_map.end(), 0);
  index_perm = psi->GetIndexPermutation();
  transform(index_perm.begin(), index_perm.end(), index_map.begin(),
	    [this,&index_tmp](const size_t i){ return index_tmp[i]; });
  T->PermuteRows(index_perm);
  delta->PermuteRows(index_perm);
  X->PermuteRows(index_perm);
    // evaluate inverse and sqrt
  auto t2 = high_resolution_clock::now();
  if(verbose) cout << "inverting input psi...";
  psiinv_in = new BGMatrix<double>("psiinv_in", *psi_in);
  psi_in->Inverse(psiinv_in);
  auto t3 = high_resolution_clock::now();
  if(verbose)
    cout << scientific << setprecision(4) <<
      duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl
	 << "inverting pairwise psi...";
  psiinv = new BGMatrix<double>("psiinv", *psi);
  psi->Inverse(psiinv);
  t2 = high_resolution_clock::now();
  if(verbose)
    cout << duration_cast<nanoseconds>(t2-t3).count()*1.0e-9 << " s" << endl
	 << "evaluating sqrt of pairwise psi...";
  psi->ChangeDevLock(true);
  sqrtpsi = psi->Sqrt();
  sqrtpsi->CloseToAppend();
  psi->ChangeDevLock(false);
  t3 = high_resolution_clock::now();
  if(verbose)
    cout << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl;
  // create block matrices with fixed block size
  offset_cbs.resize(gauss_hermite.size(), 0);
  boffset_cbs.resize(gauss_hermite.size(), 0);
  psiinv_cbs.resize(gauss_hermite.size(), NULL);
  sqrtpsi_cbs.resize(gauss_hermite.size(), NULL);
  Ebb_cbs.resize(gauss_hermite.size(), NULL);
  delta_cbs.resize(gauss_hermite.size(), NULL);
  aq_cbs.resize(gauss_hermite.size(), NULL);
  buff0_cbs.resize(gauss_hermite.size(), NULL);
  buff1_cbs.resize(gauss_hermite.size(), NULL);
  dtpb_cbs.resize(gauss_hermite.size(), NULL);
  atpb_cbs.resize(gauss_hermite.size(), NULL);
  sf0_cbs.resize(gauss_hermite.size(), NULL);
  size_t bindex = 0;
  offset = 0;
  psiinv->ChangeDevLock(true);
  sqrtpsi->ChangeDevLock(true);
  PMatrix<double>* tmpbl = new PMatrix<double>();
  const size_t N = psiinv_cbs.size();
  for(size_t i=0; i<N; i++){
    const size_t n = N - i;
    offset_cbs[i] = offset;
    boffset_cbs[i] = bindex;
    psiinv_cbs[i]  = new BGMatrix<double>("psiinv_cbs_"+to_string(n), handle);
    sqrtpsi_cbs[i] = new BGMatrix<double>("sqrtpsi_cbs_"+to_string(n), handle);
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
    psiinv_cbs[i]->CloseToAppend();
    sqrtpsi_cbs[i]->CloseToAppend();
    Ebb_cbs[i] = new BGMatrix<double>("Ebb_cbs_"+to_string(n), *sqrtpsi_cbs[i]);
    Ebb_cbs[i]->Zero();
    sqrtpsi_cbs[i]->ChangeDevLock(false);
    Ebb_cbs[i]->ChangeDevLock(false);
    delta_cbs[i] = new GMatrix<double>("delta_cbs_"+to_string(n),
				       sqrtpsi_cbs[i]->GetNRows(),
				       1, false, handle);
    for(size_t j=0; j<delta_cbs[i]->GetNData(); j++)
      delta_cbs[i]->SetVal(j, delta->GetVal(offset_cbs[i]+j));
    aq_cbs[i] = new GMatrix<double>("aq_cbs_"+to_string(n),
				    sqrtpsi_cbs[i]->GetNRows(),
				    1, false, handle);
    buff0_cbs[i] = new GMatrix<double>("buff0_cbs_"+to_string(n),
				       sqrtpsi_cbs[i]->GetNRows(),
				       1, false, handle);
    buff1_cbs[i] = new GMatrix<double>("buff1_cbs_"+to_string(n),
				       sqrtpsi_cbs[i]->GetNRows(),
				       1, false, handle);
    cudaMalloc((void**)dtpb_cbs[i],sizeof(double)*sqrtpsi_cbs[i]->GetNBlocks());
    cudaMalloc((void**)atpb_cbs[i],sizeof(double)*sqrtpsi_cbs[i]->GetNBlocks());
    cudaMalloc((void**) sf0_cbs[i],sizeof(double)*sqrtpsi_cbs[i]->GetNBlocks());
    psiinv_cbs[i]->SetUseCblas(false);
    sqrtpsi_cbs[i]->SetUseCblas(false);
    Ebb_cbs[i]->SetUseCblas(false);
    delta_cbs[i]->SetUseCblas(false);
    aq_cbs[i]->SetUseCblas(false);
    buff0_cbs[i]->SetUseCblas(false);
    buff1_cbs[i]->SetUseCblas(false);
  }
  delete tmpbl;
  psiinv->ChangeDevLock(false);
  sqrtpsi->ChangeDevLock(false);
  // initialize parameters
  niterations   = 0;
  epsilon       = 1.0e-5;
  tau           = 0.0;
  tauprev       = 0.0;
  Utau          = 0.0;
  Vtau          = 0.0;
    // initialize matrices
  ebx_in        = new GMatrix<double>("ebx_in",        nind, 1, false, handle);
  ebxw1_in      = new GMatrix<double>("ebxw1_in",      nind, 1, false, handle);
  w1_in         = new GMatrix<double>("w1_in",         nind, 1, false, handle);
  lambda_in     = new GMatrix<double>("lambda_in",     nind, 1, false, handle);
  lambdaprev_in = new GMatrix<double>("lambdaprev_in", nind, 1, false, handle);
  clambda_in    = new GMatrix<double>("clambda_in",    nind, 1, false, handle);
  b_in          = new GMatrix<double>("b_in",          nind, 1, false, handle);
  bprev_in      = new GMatrix<double>("bprev_in",      nind, 1, false, handle);
  ub_in         = new GMatrix<double>("ub_in",         nind, 1, false, handle);
  mu_in         = new GMatrix<double>("mu_in",         nind, 1, false, handle);
  sigma_in      = new GMatrix<double>("sigma_in",      nind, 1, false, handle);
  buffvec_in    = new GMatrix<double>("buffvec_in",    nind, 1, false, handle);
  buffmat_in    = new BGMatrix<double>("buffmat_in", *psi_in);
  buffmat_in->Zero();
  T_in_sort     = new GMatrix<double>("T_in_sort",     nind, 1, false, handle);
  X_in_sort     = new GMatrix<double>("X_in_sort",     nind, npar,false,handle);
  delta_in_sort = new GMatrix<double>("delta_in_sort", nind, 1, false, handle);
  ebxw1_in_sort = new GMatrix<double>("ebxw1_in_sort", nind, 1, false, handle);
  aq            = new GMatrix<double>("aq",   T->GetNRows(), 1, false, handle);
  w1q           = new GMatrix<double>("w1q",  T->GetNRows(), 1, false, handle);
  beta          = new PMatrix<double>("beta",          1, npar, false);
  betaprev      = new PMatrix<double>("betaprev",      1, npar, false);
  Ubeta         = new PMatrix<double>("Ubeta",         1, npar, false);
  Vbeta         = new PMatrix<double>("Vbeta",      npar, npar, false);
  ebx           = new GMatrix<double>("ebx",  T->GetNRows(), 1, false, handle);
  ebxw1q        = new GMatrix<double>("ebxw1q",T->GetNRows(),1, false, handle);
  lambda        = new PMatrix<double>("lambda",     T->GetNRows(), 1,    false);
  lambdaprev    = new PMatrix<double>("lambdaprev", T->GetNRows(), 1,    false);
  clambda       = new GMatrix<double>("clambda",
				      T->GetNRows(), 1, false, handle);
  T_sort        = new PMatrix<double>("T_sort",     T->GetNRows(), 1,    false);
  X_sort        = new PMatrix<double>("X_sort",     T->GetNRows(), npar, false);
  delta_sort    = new PMatrix<double>("delta_sort", T->GetNRows(), 1,    false);
  ebxw1q_sort   = new PMatrix<double>("ebxw1q_sort",T->GetNRows(), 1,    false);
  X->SetUseCblas(false);
  X_sort->SetUseCblas(false);
  X_in->SetUseCblas(false);
  X_in_sort->SetUseCblas(false);
  Ubeta->SetUseCblas(false);
  Vbeta->SetUseCblas(false);
  auto t1 = high_resolution_clock::now();
  titer_us["InitializeAP"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


GHAModelGPU::~GHAModelGPU(){
  for(auto p : gh_roots_sqrt2) if(p) cudaFree(p);
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
  if(T)             delete T;
  if(delta)         delete delta;
  if(psi)           delete psi;
  if(psiinv)        delete psiinv;
  if(sqrtpsi)       delete sqrtpsi;
  if(aq)            delete aq;
  if(w1q)           delete w1q;
  if(beta)          delete beta;
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
  for(auto p : psiinv_cbs)  if(p) delete p;
  for(auto p : sqrtpsi_cbs) if(p) delete p;
  for(auto p : Ebb_cbs)     if(p) delete p;
  for(auto p : delta_cbs)   if(p) delete p;
  for(auto p : aq_cbs)      if(p) delete p;
  for(auto p : buff0_cbs)   if(p) delete p;
  for(auto p : buff1_cbs)   if(p) delete p;
  for(auto p : dtpb_cbs)    if(p) cudaFree(p);
  for(auto p : atpb_cbs)    if(p) cudaFree(p);
  for(auto p :  sf0_cbs)    if(p) cudaFree(p);
}


GaussHermiteND GHAModelGPU::GetGaussHermite(size_t order) const{
  if(gauss_hermite.size() == 0) return GaussHermiteND();
  else if(order >= gauss_hermite.size()) return gauss_hermite[0];
  else return gauss_hermite[order];
}


void GHAModelGPU::SetSeedsAP(){
  auto t0 = high_resolution_clock::now();
  assert(epsilon > 0.0 && epsilon < 1.0);
  // reset timers
  for(auto& t : titer_us) if(t.first != "Initialize") t.second = 0.0;
  // zero parameters
  niterations = 0;
  nll         = 0.0;
  tau         = 1.0;
  tauprev     = 0.0;
  Utau        = 0.0;
  Vtau        = 0.0;
  beta->Zero();
  ebx->SetAll(1.0);
  aq->CopyToDevice();
  w1q->CopyToDevice();
  ebx->CopyToDevice();
  ebxw1q->CopyToDevice();
  clambda->CopyToDevice();
  TimeSortDataAP();
  // initialize lambda
  double dsum = accumulate(delta_sort->GetDataConst().begin(),
			   delta_sort->GetDataConst().end(), 0.0);
  const size_t n = delta_sort->GetNData();
  for(size_t i=0; i<n; i++)
    lambda->SetVal(i, delta_sort->GetVal(i) / dsum);
  for(size_t i=0; i<delta_cbs.size(); i++){
    delta_cbs[i]->CopyToDevice();
    aq_cbs[i]->CopyToDevice();
    buff0_cbs[i]->CopyToDevice();
    buff1_cbs[i]->CopyToDevice();
  }

  UpdateAP_clambda();
  UpdateAP_aq();
  auto t1 = high_resolution_clock::now();
  titer_us["SetSeedsAP"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModelGPU::TimeSortDataAP(){
  auto t0 = high_resolution_clock::now();
  if(T_perm.size() != T->GetNData()){
    T_perm.resize(T->GetNData());
    iota(T_perm.begin(), T_perm.end(), 0);
    sort(T_perm.begin(), T_perm.end(),
	 [this](size_t i, size_t j){ return T->GetVal(i) > T->GetVal(j); });
    T_perm_t.resize(T->GetNData());
    for(size_t i=0; i<T_perm.size(); i++) T_perm_t[T_perm[i]] = i;
    copy(T->GetDataConst().begin(), T->GetDataConst().end(),
	 T_sort->GetData().begin());
    T_sort->Permute(T_perm);
    copy(X->GetDataConst().begin(), X->GetDataConst().end(),
	 X_sort->GetData().begin());
    X_sort->PermuteRows(T_perm);
    copy(delta->GetDataConst().begin(), delta->GetDataConst().end(),
	 delta_sort->GetData().begin());
    delta_sort->Permute(T_perm);
  }
  ebxw1q->ReadFromDevice();
  copy(ebxw1q->GetDataConst().begin(), ebxw1q->GetDataConst().end(),
       ebxw1q_sort->GetData().begin());
  ebxw1q_sort->Permute(T_perm);
  auto t1 = high_resolution_clock::now();
  titer_us["TimeSortDataAP"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModelGPU::UpdateAP_aq(){
  auto t0 = high_resolution_clock::now();
  ComponentProduct(clambda->GetDeviceData(), ebx->GetDeviceData(),
		   aq->GetDeviceData(), aq->GetNData());
  for(size_t i=0; i<aq_cbs.size(); i++)
    cudaMemcpy(aq_cbs[i]->GetDeviceData(), aq->GetDeviceData()+offset_cbs[i],
	       sizeof(double)*aq_cbs[i]->GetNData(), cudaMemcpyDeviceToDevice);
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_aq"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModelGPU::UpdateAP_ebxw1q(){
  auto t0 = high_resolution_clock::now();
  ComponentProduct(ebx->GetDeviceData(), w1q->GetDeviceData(),
		   ebxw1q->GetDeviceData(), ebxw1q->GetNData());
  TimeSortDataAP();
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_ebxw1q"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModelGPU::UpdateAP_w1q(){
  auto t0 = high_resolution_clock::now();
  w1q->Zero();
  const double stau  = sqrt(tau);
  const double alpha = 1.0;
  for(size_t i=0; i<gauss_hermite.size(); i++){
    GaussHermiteND gh = gauss_hermite[i];
    const size_t p = gh.GetPowers()[0];
    const size_t n = gh.GetDimension();
    const size_t N = delta_cbs[i]->GetNData();
    BGMatrix<double>* spsi = sqrtpsi_cbs[i];
    GMatrix<double>* d = delta_cbs[i];
    GMatrix<double>* a = aq_cbs[i];
    GMatrix<double>* b0 = buff0_cbs[i];
    GMatrix<double>* b1 = buff1_cbs[i];
    double* dtpb = dtpb_cbs[i];
    double* atpb = atpb_cbs[i];
    double*  sf0 =  sf0_cbs[i];
    cudaMemset(dtpb, 0, sizeof(double)*spsi->GetNBlocks());
    cudaMemset(atpb, 0, sizeof(double)*spsi->GetNBlocks());
    cudaMemset(sf0,  0, sizeof(double)*spsi->GetNBlocks());
    size_t offset = offset_cbs[i];
    for(size_t j=0; j<p; j++){
      CyclicPerm(gh_roots_sqrt2[i], gh_powers[i], b0->GetDeviceData(), j, n, N);
      cudaDeviceSynchronize();
      spsi->RightMult(*b0, stau, b1, 0.0);
      ComponentExponential(b1->GetDeviceData(), b0->GetDeviceData(), N);
      InnerProdStride(d->GetDeviceData(), b1->GetDeviceData(),
		      dtpb, n, spsi->GetNBlocks());
      InnerProdStride(a->GetDeviceData(), b0->GetDeviceData(),
		      atpb, n, spsi->GetNBlocks());
      DifferenceExponential(dtpb, atpb, gh.GetWeightProd(j),spsi->GetNBlocks());
      cublasDaxpy(handle, spsi->GetNBlocks(), &alpha, dtpb, 1, sf0, 1);
      ExpModProduct(w1q->GetDeviceData() + offset,
		    b1->GetDeviceData(), dtpb, n, N);
    }
    DivideMod(w1q->GetDeviceData() + offset, sf0, n, N);
  }
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_w1q"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;  
}


void GHAModelGPU::UpdateAP_beta(){
  auto t0 = high_resolution_clock::now();
  betaprev->SetData(beta->GetData());
  Ubeta->Zero();
  Vbeta->Zero();
  size_t j = 0;
  double tmp = 0.0, tmpval = 0.0;
  PMatrix<double>* tmprow = new PMatrix<double>("tmprow", *Ubeta);
  PMatrix<double>* tmpmat = new PMatrix<double>("tmpmat", *Vbeta);
  tmprow->SetUseCblas(false); tmpmat->SetUseCblas(false);
  const vector<double> t = T_sort->GetDataConst();
  const vector<double> d = delta_sort->GetDataConst();
  const vector<double> ew = ebxw1q_sort->GetDataConst();
  const size_t n = T_sort->GetNData();
  for(size_t i=0; i<n; ){
    while(i == j || (i > 0 ? t[i-1] == t[i] : true)){
      tmp = ew[i];
      tmpval += tmp;
      tmprow->AddMultRow(0, i, tmp, *X_sort);
      X_sort->OuterRowProduct(i, i, tmp, tmpmat);
      i ++;
      if(i == n) break;
    }
    for(; j<i; j++){
      if(d[j] == 0.0) continue;
      Ubeta->AddMultRow(0, j, 1.0, *X_sort);
      Ubeta->AddMultRow(0, 0, -1/tmpval, *tmprow);
      Vbeta->axpy(1/tmpval, *tmpmat);
      tmprow->OuterRowProduct(0, 0, -1/tmpval/tmpval, Vbeta);
    }
  }
  if(tmpmat) delete tmpmat;
  tmpmat = Vbeta->Inverse();
  Ubeta->RightMult(*tmpmat, tmprow);
  (*beta) += (*tmprow);
  double* beta_dev = NULL;
  cudaMalloc((void**) &beta_dev, sizeof(double)*beta->GetNData());
  cudaMemcpy(beta_dev, &beta->GetDataConst().front(),
	     sizeof(double)*beta->GetNData(), cudaMemcpyDeviceToHost);
  InnerRowProd(X->GetDeviceData(), beta_dev, ebx->GetDeviceData(),
	       beta->GetNData(), ebx->GetNData());
  cudaFree(beta_dev);
  if(tmprow) delete tmprow;
  if(tmpmat) delete tmpmat;
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_beta"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModelGPU::UpdateAP_lambda(){
  auto t0 = high_resolution_clock::now();
  lambdaprev->SetData(lambda->GetData());
  lambda->Zero();
  size_t j = 0;
  double tmp = 0.0;
  const vector<double> t = T_sort->GetDataConst();
  const vector<double> d = delta_sort->GetDataConst();
  const vector<double> ew = ebxw1q_sort->GetDataConst();
  const size_t n = T_sort->GetNData();
  for(size_t i=0; i<n; ){
    while(i == j || (i > 0 ? t[i-1] == t[i] : true)){
      tmp += ew[i];
      i ++;
      if(i == n) break;
    }
    for(; j<i; j++) lambda->SetVal(j, d[j]/tmp);
  }
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_lambda"]+=duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GHAModelGPU::UpdateAP_clambda(){
  auto t0 = high_resolution_clock::now();
  clambda->Zero();
  double tmp = 0.0;
  const vector<double> t = T_sort->GetDataConst();
  const vector<double> d = delta_sort->GetDataConst();
  const vector<double> ew = ebxw1q_sort->GetDataConst();
  const size_t n = T_sort->GetNData();
  size_t j = n - 1;
  for(size_t i=n-1; (i>=0 && i<n) || i==(size_t)(-1); ){
    while(i == j || (i<n-1 ? t[i+1]==t[i] : true)){
      tmp += lambda->GetVal(i);
      i --;
      if(i > n) break;
    }
    for(; i>n?j>=0:j>i; j--){
      clambda->SetVal(j, tmp);
      if(j == 0) break;
    }
  }
  clambda->Permute(T_perm_t);
  clambda->CopyToDevice();
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_clambda"]+=duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
}


void GHAModelGPU::UpdateAP_tau(){
  auto t0 = high_resolution_clock::now();
  const double stau = sqrt(tau);
  const double alpha = 1.0;
  tauprev = tau;
  tau = 0.0;
  for(size_t i=0; i<gauss_hermite.size(); i++){
    GaussHermiteND gh = gauss_hermite[i];
    const size_t p = gh.GetPowers()[0];
    const size_t n = gh.GetDimension();
    const size_t N = delta_cbs[i]->GetNData();
    BGMatrix<double>* spsi = sqrtpsi_cbs[i];
    BGMatrix<double>* ebb = Ebb_cbs[i];
    ebb->Zero();
    GMatrix<double>* d  = delta_cbs[i];
    GMatrix<double>* a  = aq_cbs[i];
    GMatrix<double>* b0 = buff0_cbs[i];
    GMatrix<double>* b1 = buff1_cbs[i];
    double* dtpb = dtpb_cbs[i];
    double* atpb = atpb_cbs[i];
    double*  sf0 =  sf0_cbs[i];
    cudaMemset(dtpb, 0, sizeof(double)*spsi->GetNBlocks());
    cudaMemset(atpb, 0, sizeof(double)*spsi->GetNBlocks());
    cudaMemset(sf0,  0, sizeof(double)*spsi->GetNBlocks());
    for(size_t j=0; j<p; j++){
      CyclicPerm(gh_roots_sqrt2[i], gh_powers[i], b0->GetDeviceData(), j, n, N);
      spsi->RightMult(*b0, stau, b1, 0.0);
      ComponentExponential(b1->GetDeviceData(), b0->GetDeviceData(), N);
      InnerProdStride(d->GetDeviceData(), b1->GetDeviceData(), dtpb, n, N);
      InnerProdStride(a->GetDeviceData(), b0->GetDeviceData(), atpb, n, N);
      DifferenceExponential(dtpb, atpb, gh.GetWeightProd(j),spsi->GetNBlocks());
      cublasDaxpy(handle, spsi->GetNBlocks(), &alpha, dtpb, 1, sf0, 1);
      OuterModProd(b1->GetDeviceData(), dtpb,
		   ebb->GetDeviceData(), n, ebb->GetNData());
    }
    DivideMod(ebb->GetDeviceData(), sf0, n*n, ebb->GetNData());
    ebb->ChangeDevLock(true);
    tau += psiinv_cbs[i]->TraceProduct(*ebb);
    ebb->ChangeDevLock(false);
  }
  tau /= psi->GetNRows();
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateAP_tau"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


bool GHAModelGPU::IterateAP(size_t niter){
  for(size_t i=0; i<niter; i++){
    auto t0 = high_resolution_clock::now();
    niterations ++;
    UpdateAP_w1q();
    UpdateAP_ebxw1q();
    UpdateAP_beta();
    UpdateAP_ebxw1q();
    UpdateAP_lambda();
    UpdateAP_clambda();
    UpdateAP_aq();
    UpdateAP_tau();
    auto t1 = high_resolution_clock::now();
    titer_us["IterateAP"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    if(verbose)
      cout << scientific << "iteration " << niterations << " "<< setprecision(3)
	   << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us:" <<endl;
    bool converge = CheckConvergenceAP();
    if(converge) return true;
  }
  return false;
}


bool GHAModelGPU::CheckConvergenceAP(){
  auto t0 = high_resolution_clock::now();
  double bnorm = 0.0;
  for(size_t i=0; i<npar; i++)
    bnorm += abs(beta->GetVal(i) - betaprev->GetVal(i));
  double lnorm = 0.0;
  for(size_t i=0; i<nind; i++)
  lnorm = std::max(lnorm, abs(lambda->GetVal(i)-lambdaprev->GetVal(i)));
  double tnorm = abs(tau - tauprev);
  if(verbose){
    beta->Print(beta->GetNData(), 4, "    ");
    cout << "    tau = " << tau << endl << "    "
	 << "||beta|| = "   << setprecision(3)   << bnorm << ", "
	 << "||lambda|| = " << setprecision(3)   << lnorm << ", "
	 << "|tau| = "      << setprecision(3)   << tnorm << endl;
  }
  auto t1 = high_resolution_clock::now();
  titer_us["CheckConvergenceAP"]=duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
  if(std::max(tnorm, std::max(bnorm, lnorm)) < epsilon) return true;
  return false;
}


void GHAModelGPU::Solve(size_t maxiterap, size_t maxiterin){
  auto t0 = high_resolution_clock::now();
  SetSeedsAP();
  while(!IterateAP(1) &&
	(maxiterap > 0 ? niterations < maxiterap : true)){ }
  auto t1 = high_resolution_clock::now();
  cout << endl << "Step 1 solution after " << niterations << " iterations, "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << " s:" << endl;
  beta->Print(beta->GetNData(), 4, "    ");
  cout << "tau = " << tau << endl << endl
       << "Fixing tau, iterating for step 2..." << endl;
  for(auto& pr : titer_us)
    if(pr.second != 0.0){
      cout << pr.first << ":  " << pr.second << " us  "
	   << pr.second/niterations << " us/iter" << endl;
      pr.second = 0.0;
    }
  /*t0 = high_resolution_clock::now();
  SetSeedsIn();
  while(!IterateIn(1) &&
	(maxiterin > 0 ? niterations < maxiterin : true)){ }
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
      }*/
}

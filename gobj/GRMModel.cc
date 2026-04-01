#include "GRMModel.hh"
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

double BlockDiagInverse(PMatrix<double>* A, size_t nbl, size_t bsize){
  double logdet = 0.0;
  assert(A->GetNRows() == A->GetNCols());
  if(bsize == 2){
    assert(A->GetNRows() % 2 == 0);
    double det;
    size_t j = 1;
    double tmp;
    for(size_t i=0; i<A->GetNRows(); (i+=2, j+=2)){
      det = (A->GetVal(i, i) * A->GetVal(j, j) -
	     A->GetVal(i, j) * A->GetVal(j, i));
      tmp = A->GetVal(i, i) / det;
      A->SetVal(i, i,  A->GetVal(j, j) / det);
      A->SetVal(j, j,  tmp);
      A->SetVal(i, j, -A->GetVal(i, j) / det);
      A->SetVal(j, i,  A->GetVal(i, j));
      logdet += log(abs(det));
    }
  }
  else{
    PMatrix<double>* tmp = new PMatrix<double>("tmp", bsize, bsize, false);
    const size_t nindbl = nbl * bsize;
    for(size_t i=0; i<nindbl; i+=bsize){
      A->GetBlock(i, i, tmp);
      tmp->Invert(&logdet);
      A->SetBlock(i, i, *tmp);
    }
    for(size_t i=nindbl; i<A->GetNRows(); i++)
      A->SetVal(i, i, 1 / A->GetVal(i, i));
    delete tmp;
  }
  return logdet;
}


/*void GRMModel::tpiU_inverse(){
  // copy -1/tau and ub to device
  auto t0 = high_resolution_clock::now();
  *tau_host = -1 / tau;
  ub->PermuteRows(permutationt);
  cudaMemcpy(ub_dev, &ub->GetData().front(),
	     5000*sizeof(double), cudaMemcpyHostToDevice);
  // copy psiinv to tmp and scale by -1/tau
  auto t1 = high_resolution_clock::now();
  cudaMemcpy(tmp_dev, psiinv_dev,
	     tpiU_host.size()*sizeof(double), cudaMemcpyDeviceToDevice);
  cublasDscal(handle, tpiU_host.size(), tau_host, tmp_dev, 1);
  // add -ub to tpiU
  auto t2 = high_resolution_clock::now();
  *alpha_host = -1.0;
  for(size_t i=0; i<500; i++)
    cublasDaxpy(handle, 10, alpha_host, ub_dev+i*10, 1, tmp_dev+i*100, 11);
  // batched inverse of tpiU
  auto t3 = high_resolution_clock::now();
  cudaMemset(tpiU_dev, 0, tpiU_host.size()*sizeof(double));
  cublasDgetrfBatched(handle, 10, pntrs_dev, 10, pivot_dev, info_dev, 500);
  cublasDgetriBatched(handle, 10, pntrs_dev, 10, pivot_dev,
		      ipntrs_dev, 10, info_dev, 500);
  cudaMemcpy(&tpiU_host.front(), tpiU_dev,
	     tpiU_host.size()*sizeof(double), cudaMemcpyDeviceToHost);
  auto t4 = high_resolution_clock::now();
  tpiU->Zero();
  size_t l = 0;
  for(size_t i=0; i<5000; i+=10)
    for(size_t j=0; j<10; j++)
      for(size_t k=0; k<10; (k++, l++))
	tpiU->SetVal(permutationt[i+j], permutationt[i+k], tpiU_host[l]);
  for(size_t i=5000; i<tpiU->GetNRows(); i++)
    tpiU->SetVal(permutationt[i], permutationt[i],
		 -1 / (psiinvun->GetVal(i, i)/tau + ub->GetVal(i)));
  ub->PermuteRows(permutation);
  cout << duration_cast<nanoseconds>(t1-t0).count()*1.0e-9 << "  "
       << duration_cast<nanoseconds>(t2-t1).count()*1.0e-9 << "  "
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << "  "
       << duration_cast<nanoseconds>(t4-t3).count()*1.0e-9 << endl;
       }*/


GRMModel::GRMModel(PMatrix<double>* t, PMatrix<double>* d,
		   PMatrix<double>* x, PMatrix<double>* p,
		   PMatrix<double>* pinv){
  // setup clocking
  vector<string> parts({
      "Initialize", "ReadData", "TimeSortData", "SetSeeds",
      "Updateb", "UpdateLambda","UpdateTau", "GSTau", "MHTau",
      "CheckConvergence", "Total"});
  for(auto const& p : parts){
    titer_us[p]  = 0.0;
    ttotal_us[p] = 0.0;
  }
  auto t0 = high_resolution_clock::now();
  // check dimensions
  assert(t->GetNRows() == d->GetNRows());
  assert(t->GetNCols() == 1 && d->GetNCols() == 1);
  assert(x->GetNRows() == t->GetNRows());
  assert(p->GetNRows() == x->GetNRows() && p->GetNCols() == x->GetNRows());
  if(pinv)
    assert(pinv->GetNRows()==p->GetNRows() && pinv->GetNCols()==p->GetNCols());
  nind = t->GetNRows();
  npar = x->GetNCols();
  verbose           = true;
  b_partial_laplace = true;
  b_from_last       = true;
  T     = t;
  delta = d;
  X     = x;
  psi   = p;
  psilogdet  = 0.0;
  tpiUlogdet = 0.0;
  if(pinv){
    psiinv = pinv;
    psiinvun = new PMatrix<double>("psiinvun", *psiinv);
    psiinv->Permute(permutation);
    delpsiinv = false;
  }
  else{
    auto t2 = high_resolution_clock::now();
    if(verbose) cout << "inverting psi...";
    psiinv = new PMatrix<double>(psi->GetName()+"_inv", *psi);
    //psi->BlockwiseInverse(psiinv, psi->GetNRows()/64, &psilogdet);
    psilogdet = BlockDiagInverse(psiinv, 500, 10);
    //psilogdet = BlockDiagInverse(psiinv, 400, 3);
    //psilogdet = BlockDiagInverse(psiinv, 0, 2);
    psiinvun = new PMatrix<double>("psiinvun", *psiinv);
    auto t3 = high_resolution_clock::now();
    if(verbose)
      cout << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << " s" << endl;
    delpsiinv = true;
  }
  // gpu setup
  tau_host   = (double*) malloc(sizeof(double));
  alpha_host = (double*) malloc(sizeof(double));
  gamma_host = (double*) malloc(sizeof(double));
  tpiU_host.resize(50000);
  cublasCreate_v2(&handle);
  cudaMalloc((void**) &b_dev,            sizeof(double)*5000);
  cudaMalloc((void**) &ub_dev,           sizeof(double)*5000);
  cudaMalloc((void**) &mu_dev,           sizeof(double)*5000);
  cudaMalloc((void**) &w_dev,            sizeof(double)*5000);
  cudaMalloc((void**) &delta_dev,        sizeof(double)*5000);
  cudaMalloc((void**) &psiinv_dev,       sizeof(double)*tpiU_host.size());
  cudaMalloc((void**) &tmp_dev,          sizeof(double)*tpiU_host.size());
  cudaMalloc((void**) &tpiU_dev,         sizeof(double)*tpiU_host.size());
  cudaMalloc((void**) &b_pntrs_dev,      sizeof(double*)*500);
  cudaMalloc((void**) &mu_pntrs_dev,     sizeof(double*)*500);
  cudaMalloc((void**) &w_pntrs_dev,      sizeof(double*)*500);
  cudaMalloc((void**) &psiinv_pntrs_dev, sizeof(double*)*500);
  cudaMalloc((void**) &tmp_pntrs_dev,    sizeof(double*)*500);
  cudaMalloc((void**) &tpiU_pntrs_dev,   sizeof(double*)*500);
  cudaMalloc((void**) &pivot_dev,        sizeof(int)*5000);
  cudaMalloc((void**) &info_dev,         sizeof(int)*500);
  double** b_pntrs_host      = (double**) malloc(sizeof(double*)*500);
  double** mu_pntrs_host     = (double**) malloc(sizeof(double*)*500);
  double** w_pntrs_host      = (double**) malloc(sizeof(double*)*500);
  double** psiinv_pntrs_host = (double**) malloc(sizeof(double*)*500);
  double** tmp_pntrs_host    = (double**) malloc(sizeof(double*)*500);
  double** tpiU_pntrs_host   = (double**) malloc(sizeof(double*)*500);
  for(size_t i=0; i<500; i++){
    b_pntrs_host[i]      = b_dev      + i*10;
    mu_pntrs_host[i]     = mu_dev     + i*10;
    w_pntrs_host[i]      = w_dev      + i*10;
    psiinv_pntrs_host[i] = psiinv_dev + i*100;
    tmp_pntrs_host[i]    = tmp_dev    + i*100;
    tpiU_pntrs_host[i]   = tpiU_dev   + i*100;
  }
  cudaMemcpy(b_pntrs_dev, b_pntrs_host,
	     sizeof(double*)*500, cudaMemcpyHostToDevice);
  cudaMemcpy(mu_pntrs_dev, mu_pntrs_host,
	     sizeof(double*)*500, cudaMemcpyHostToDevice);
  cudaMemcpy(w_pntrs_dev, w_pntrs_host,
	     sizeof(double*)*500, cudaMemcpyHostToDevice);
  cudaMemcpy(psiinv_pntrs_dev, psiinv_pntrs_host,
	     sizeof(double*)*500, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_pntrs_dev, tmp_pntrs_host,
	     sizeof(double*)*500, cudaMemcpyHostToDevice);
  cudaMemcpy(tpiU_pntrs_dev, tpiU_pntrs_host,
	     sizeof(double*)*500, cudaMemcpyHostToDevice);
  free(b_pntrs_host);
  free(mu_pntrs_host);
  free(w_pntrs_host);
  free(psiinv_pntrs_host);
  free(tmp_pntrs_host);
  free(tpiU_pntrs_host);
  cudaMemcpy(delta_dev, &delta->GetData().front(),
	     5000*sizeof(double), cudaMemcpyHostToDevice);
  size_t l = 0;
  for(size_t i=0; i<5000; i+=10)
    for(size_t j=0; j<10; j++)
      for(size_t k=0; k<10; (k++, l++))
	tpiU_host[l] = psiinv->GetVal(i+j, i+k);
  cudaMemcpy(psiinv_dev, &tpiU_host.front(),
	     tpiU_host.size()*sizeof(double), cudaMemcpyHostToDevice);
  TimeSortData();
  // initialize parameters
  gauss_hermite = GaussHermite2D("/home/tcald/bios/hermite/data","hermite",20);
  niterations   = 0;
  epsilon       = 1.0e-6;
  tau           = 0.0;
  tauprev       = 0.0;
  Utau          = 0.0;
  Vtau          = 0.0;
  // initialize matrices
  beta       = new PMatrix<double>("beta",       1,    npar, false);
  betaprev   = new PMatrix<double>("betaprev",   1,    npar, false);
  Ubeta      = new PMatrix<double>("Ubeta",      1,    npar, false);
  Vbeta      = new PMatrix<double>("Vbeta",      npar, npar, false);
  ebx        = new PMatrix<double>("ebx",        nind, 1   , false);
  b          = new PMatrix<double>("b",          nind, 1   , false);
  bprev      = new PMatrix<double>("bprev",      nind, 1   , false);
  ub         = new PMatrix<double>("ub",         nind, 1   , false);
  lambda     = new PMatrix<double>("lambda",     nind, 1   , false);
  lambdaprev = new PMatrix<double>("lambdaprev", nind, 1   , false);
  clambda    = new PMatrix<double>("clambda",    nind, 1   , false);
  w          = new PMatrix<double>("w",          nind, 1   , false);
  mu         = new PMatrix<double>("mu",         nind, 1   , false);
  sigma2     = new PMatrix<double>("sigma2",     nind, 1   , false);
  buffmat    = new PMatrix<double>("buffmat",    nind, nind, false);
  tpiU       = NULL;
  auto t1 = high_resolution_clock::now();
  ttotal_us["Initialize"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


GRMModel::~GRMModel(){
  if(beta)       delete beta;
  if(betaprev)   delete betaprev;
  if(Ubeta)      delete Ubeta;
  if(Vbeta)      delete Vbeta;
  if(ebx)        delete ebx;
  if(b)          delete b;
  if(bprev)      delete bprev;
  if(ub)         delete ub;
  if(lambda)     delete lambda;
  if(lambdaprev) delete lambdaprev;
  if(clambda)    delete clambda;
  if(w)          delete w;
  if(mu)         delete mu;
  if(sigma2)     delete sigma2;
  if(buffmat)    delete buffmat;
  if(tpiU)       delete tpiU;
  if(psiinvun)   delete psiinvun;
  if(delpsiinv && psiinv) delete psiinv;
  free(tau_host);
  free(alpha_host);
  free(gamma_host);
  cudaFree(b_dev);
  cudaFree(ub_dev);
  cudaFree(mu_dev);
  cudaFree(w_dev);
  cudaFree(delta_dev);
  cudaFree(psiinv_dev);
  cudaFree(tmp_dev);
  cudaFree(tpiU_dev);
  cudaFree(b_pntrs_dev);
  cudaFree(mu_pntrs_dev);
  cudaFree(w_pntrs_dev);
  cudaFree(psiinv_pntrs_dev);
  cudaFree(tmp_pntrs_dev);
  cudaFree(tpiU_pntrs_dev);
  cudaFree(pivot_dev);
  cudaFree(info_dev);
  cublasDestroy_v2(handle);
}


void GRMModel::LoadGaussHermite(string path, string basefn, size_t order){
  gauss_hermite = GaussHermite2D(path, basefn, order);
}


void GRMModel::TimeSortData(){
  auto t0 = high_resolution_clock::now();
  permutation.resize(T->GetNData());
  iota(permutation.begin(), permutation.end(), 0);
  sort(permutation.begin(), permutation.end(),
       [this](size_t i, size_t j){ return T->GetVal(i) > T->GetVal(j); });
  permutationt.resize(T->GetNData());
  for(size_t i=0; i<permutation.size(); i++) permutationt[permutation[i]] = i;
  T->PermuteRows(permutation);
  X->PermuteRows(permutation);
  delta->PermuteRows(permutation);
  psi->Permute(permutation);
  psiinv->Permute(permutation);
  auto t1 = high_resolution_clock::now();
  ttotal_us["TimeSortData"] += duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
}


void GRMModel::SetSeeds(){
  auto t0 = high_resolution_clock::now();
  // make sure setup is consistent
  assert(epsilon > 0.0 && epsilon < 1.0);
  assert(gauss_hermite.GetOrder() > 0);
  // reset timers
  for(auto& t : ttotal_us) if(t.first != "Initialization") t.second = 0.0;
  // zero parameters
  niterations = 0;
  nll         = 0.0;
  tau         = 1.0;
  tauprev     = 0.0;
  Utau        = 0.0;
  Vtau        = 0.0;
  beta->Zero();
  ebx->SetAll(1.0);
  b->Zero();
  mu->Zero();
  // initialize sigma2, w, lambda, and ub
  double dsum = accumulate(delta->GetDataConst().begin(),
			   delta->GetDataConst().end(), 0.0);
  for(size_t i=0; i<lambda->GetNData(); i++)
    lambda->SetVal(i, delta->GetVal(i) / dsum);
  size_t j = T->GetNData() - 1;
  double tmp = 0.0;
  for(size_t i=T->GetNData()-1; (i>=0&&i<T->GetNData())||i==(size_t)(-1); ){
    while(i == j || (i<T->GetNData()-1 ? T->GetVal(i+1)==T->GetVal(i) : true)){
      tmp += lambda->GetVal(i);
      i --;
      if(i > T->GetNData()) break;
    }
    for(; i>T->GetNData()?j>=0:j>i; j--){
      clambda->SetVal(j, tmp);
      if(j == 0) break;
    }
  }
  for(size_t i=0; i<lambda->GetNData(); i++){
    ub->SetVal(i, clambda->GetVal(i));
    sigma2->SetVal(i, 1/psiinv->GetVal(i, i));
    if(b_partial_laplace){
      double s0 = 0.0, s1 = 0.0;
      for(size_t j=0; j<gauss_hermite.GetOrder(); j++){
	double arg = sqrt(2*sigma2->GetVal(i)) * gauss_hermite.GetRoot(j);
	double m = delta->GetVal(i) * arg;
	m = exp(m - clambda->GetVal(i) * ebx->GetVal(i) * exp(arg));
	s0 += exp(arg) * m * gauss_hermite.GetWeight(j);
	s1 += m * gauss_hermite.GetWeight(j);
      }
      w->SetVal(i, s0 / s1);
    }
    else{
      if(i == 0){
	buffmat->Zero();
	(*buffmat) -= (*psiinv);
	(*buffmat) *= (1.0 / tau);
	if(!tpiU)
	  tpiU = new PMatrix<double>(buffmat->GetName()+"_inv", *psiinvun);
	else tpiU->SetData(psiinvun->GetData());
	(*tpiU) *= -1.0 / tau;
	for(size_t i=0; i<nind; i++){
	  //buffmat->SetVal(i, i, buffmat->GetVal(i, i) - ub->GetVal(i));
	  tpiU->SetVal(i, i, tpiU->GetVal(i, i) - ub->GetVal(permutationt[i]));
	}
	tpiUlogdet = 0.0;
	//tpiU->Permute(permutationt);
	tpiUlogdet = BlockDiagInverse(tpiU, 500, 10);
	//tpiUlogdet = BlockDiagInverse(tpiU, 400, 3);
	//tpiUlogdet = BlockDiagInverse(tpiU, 0, 2);
	tpiU->Permute(permutation);
      }
      w->SetVal(i, b->GetVal(i) - 0.5*tpiU->GetVal(i, i));
    }
  }
  auto t1 = high_resolution_clock::now();
  ttotal_us["SetSeeds"] += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GRMModel::UpdateNLL(){
  auto t0 = high_resolution_clock::now();
  nllprev = nll;
  nll = 0.5 * (psi->GetNRows()*log(tau) + psilogdet + tpiUlogdet);
  PMatrix<double>* tmp = new PMatrix<double>("tmp", *b);
  psiinv->RightMult(*b, tmp);
  btpib = 0.5 * b->InnerColProduct(0, 0, *tmp) / tau;
  nll += btpib;
  delete tmp;
  for(size_t i=0; i<psi->GetNRows(); i++){
    nll += ebx->GetVal(i) * clambda->GetVal(i) * exp(b->GetVal(i));
    if(delta->GetVal(i) != 0.0)
      nll -= log(lambda->GetVal(i)) + log(ebx->GetVal(i)) + b->GetVal(i);
  }
  auto t1 = high_resolution_clock::now();
  titer_us["NLL"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e3;
}


void GRMModel::UpdateTauGradient(){
  auto t0 = high_resolution_clock::now();
  gradprev = grad;
  grad = -btpib + 0.5 * (psi->GetNRows()+tpiU->TraceProduct(*psiinv)/tau);
  //cout << btpib/tau << " " << 0.5*psi->GetNRows()/tau << " " << 0.5*tpiU->TraceProduct(*psiinv)/tau/tau << endl;
  grad /= tau;
  //cout << grad << endl;
  auto t1 = high_resolution_clock::now();
  titer_us["Gradent"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GRMModel::UpdateTauGradient2(){
  auto t0 = high_resolution_clock::now();
  gradprev = grad;
  grad = 0.0;
  double arg0, arg1, s0, s1, s, t, m, mu0, mu1;
  PMatrix<double>* sqrtSigma = new PMatrix<double>("sigma", 2, 2, false);
  for(size_t i=0; i<psi->GetNRows(); i++){
    for(size_t j=0; j<psi->GetNRows(); j++){
      if(psiinv->GetVal(i, j) == 0.0) continue;
      s0 = 0.0; s1 = 0.0; mu0 = 0.0; mu1 = 0.0;
      if(i == j){
	for(size_t k=0; k<gauss_hermite.GetOrder(); k++){
	  arg0 = sqrt(2*sigma2->GetVal(i)) * gauss_hermite.GetRoot(k);
	  arg0 += mu->GetVal(i);
	  m  = delta->GetVal(i) * arg0;
	  m  = exp(m - clambda->GetVal(i) * ebx->GetVal(i) * exp(arg0));
	  m *= gauss_hermite.GetWeight(k);
	  s0 += arg0 * arg0 * m;
	  s1 += m;
	}
      }
      else{
	sqrtSigma->SetVal(0, 0, psiinv->GetVal(i, i) / tau);
	sqrtSigma->SetVal(0, 1, psiinv->GetVal(i, j) / tau);
	sqrtSigma->SetVal(1, 0, psiinv->GetVal(j, i) / tau);
	sqrtSigma->SetVal(1, 1, psiinv->GetVal(j, j) / tau);
	sqrtSigma->Invert();
	s = sqrt(sqrtSigma->GetVal(0, 0)*sqrtSigma->GetVal(1, 1) -
		 sqrtSigma->GetVal(0, 1)*sqrtSigma->GetVal(1, 0));
	t = sqrt(sqrtSigma->Trace() + 2*s);
	sqrtSigma->AddIdentity(s);
	(*sqrtSigma) *= 1 / t;
	for(size_t k=0; k<psi->GetNRows(); k++){
	  if(k == i || k == j) continue;
	  s0 += psiinv->GetVal(i, k) * b->GetVal(k);
	  s1 += psiinv->GetVal(j, k) * b->GetVal(k);
	  mu0 = -psiinv->GetVal(j, j)*s0 + psiinv->GetVal(i, j)*s1;
	  mu1 = -psiinv->GetVal(i, i)*s1 + psiinv->GetVal(i, j)*s0;
	  s = psiinv->GetVal(i, i)*psiinv->GetVal(j, j) -
	    psiinv->GetVal(i, j)*psiinv->GetVal(j, i);
	  mu0 /= s;
	  mu1 /= s;
	}
	s0 = 0.0; s1 = 0.0;
	for(size_t k=0; k<gauss_hermite.GetOrder(); k++){
	  for(size_t l=0; l<gauss_hermite.GetOrder(); l++){
	    arg0 =  sqrtSigma->GetVal(0, 0) * gauss_hermite.GetRoot(k);
	    arg0 += sqrtSigma->GetVal(0, 1) * gauss_hermite.GetRoot(l);
	    arg1 =  sqrtSigma->GetVal(1, 0) * gauss_hermite.GetRoot(k);
	    arg1 += sqrtSigma->GetVal(1, 1) * gauss_hermite.GetRoot(l);
	    arg0 = M_SQRT2 * arg0 + mu0;
	    arg1 = M_SQRT2 * arg1 + mu1;
	    m = delta->GetVal(i)*arg0 + delta->GetVal(j)*arg1;
	    m -= clambda->GetVal(i) * ebx->GetVal(i) * exp(arg0);
	    m -= clambda->GetVal(j) * ebx->GetVal(j) * exp(arg1);
	    m  = exp(m) * gauss_hermite.GetWeightProd(k, l);
	    s0 += arg0 * arg1 * m;
	    s1 += m;
	  }
	}
      }
      if(s0 != 0.0) s0 /= s1;
      grad += psiinv->GetVal(i, j) * s0;
    }
  }
  grad = 0.5 * (psi->GetNRows() - grad/tau) / tau;
  delete sqrtSigma;
  auto t1 = high_resolution_clock::now();
  titer_us["Gradent"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
} 


void GRMModel::Updateb(){
  auto t0 = high_resolution_clock::now();
  bprev->SetData(b->GetData());
  // update b, use w and mu as buffers since they will be updated below
  /*psiinv->RightMult(*b, w);
  (*w) *= (-1.0 / tau);
  (*w) += (*delta);
  (*w) -= (*ub);
  buffmat->Zero();
  (*buffmat) -= (*psiinv);
  (*buffmat) *= (1.0 / tau);*/

  b->PermuteRows(permutationt);
  ub->PermuteRows(permutationt);
  cudaMemcpy(b_dev, &b->GetData().front(),
	     5000*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(ub_dev, &ub->GetData().front(),
	     5000*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(w_dev, 0, 5000*sizeof(double));
  *tau_host = -1 / tau;
  *gamma_host  = 0.0;
  cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 10, 1, 10, tau_host,
		     psiinv_pntrs_dev, 10, b_pntrs_dev, 10, gamma_host,
		     w_pntrs_dev,      10, 500);
  *alpha_host = 1.0;
  cublasDaxpy(handle, 5000, alpha_host, delta_dev, 1, w_dev, 1);
  *alpha_host = -1.0;
  cublasDaxpy(handle, 5000, alpha_host, ub_dev, 1, w_dev, 1);
  
  if(!tpiU) tpiU = new PMatrix<double>(buffmat->GetName()+"_inv", *psiinvun);
  //  else tpiU->SetData(psiinvun->GetData());
  //  (*tpiU) *= -1.0 / tau;
  //  for(size_t i=0; i<nind; i++)
  //    tpiU->SetVal(i, i, tpiU->GetVal(i, i) - ub->GetVal(permutationt[i]));
  tpiUlogdet = 0.0;
  //tpiU->Permute(permutationt);
  //  tpiUlogdet = BlockDiagInverse(tpiU, 500, 10);
  //tpiUlogdet = BlockDiagInverse(tpiU, 400, 3);
  //tpiUlogdet = BlockDiagInverse(tpiU, 0, 2);
  //  tpiU->Permute(permutation);
  //buffmat->BlockwiseInverse(tpiU, buffmat->GetNRows()/64, &tpiUlogdet);

  cudaMemcpy(tmp_dev, psiinv_dev,
	     tpiU_host.size()*sizeof(double), cudaMemcpyDeviceToDevice);
  cublasDscal(handle, tpiU_host.size(), tau_host, tmp_dev, 1);
  *alpha_host = -1.0;
  for(size_t i=0; i<500; i++)
    cublasDaxpy(handle, 10, alpha_host, ub_dev+i*10, 1, tmp_dev+i*100, 11);
  cudaMemset(tpiU_dev, 0, tpiU_host.size()*sizeof(double));
  cublasDgetrfBatched(handle, 10, tmp_pntrs_dev, 10, pivot_dev, info_dev, 500);
  cublasDgetriBatched(handle, 10, tmp_pntrs_dev, 10, pivot_dev,
		      tpiU_pntrs_dev, 10, info_dev, 500);
  
  /*if(buffmat->GetNRows() % 2 != 0) tpiUlogdet *= -1;
  tpiU->RightMult(*w, mu);
  (*tpiU) *= -1.0;
  (*b) -= (*mu);*/

  *alpha_host = 1.0;
  cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 10, 1, 10, alpha_host,
		     tpiU_pntrs_dev, 10, w_pntrs_dev, 10, gamma_host,
		     mu_pntrs_dev,   10, 500);
  *alpha_host = -1.0;
  cublasDscal(handle, tpiU_host.size(), alpha_host, tpiU_dev, 1);
  cudaMemcpy(&tpiU_host.front(), tpiU_dev,
	     tpiU_host.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cublasDaxpy(handle, 5000, alpha_host, mu_dev, 1, b_dev, 1);
  cudaMemcpy(&b->GetData().front(), b_dev,
	     5000*sizeof(double), cudaMemcpyDeviceToHost);
  
  // update mu and sigma
  //for(size_t i=0; i<nind; i++)
  //buffmat->SetVal(i, i, buffmat->GetVal(i, i) + ub->GetVal(i));
  /*buffmat->RightMult(*b, mu);*/

  cudaMemcpy(tmp_dev, psiinv_dev,
	     tpiU_host.size()*sizeof(double), cudaMemcpyDeviceToDevice);
  cublasDscal(handle, tpiU_host.size(), tau_host, tmp_dev, 1);
  *alpha_host = 1.0;
  cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 10, 1, 10, alpha_host,
		     tmp_pntrs_dev, 10, b_pntrs_dev, 10, gamma_host,
		     mu_pntrs_dev,  10, 500);
  cudaMemcpy(&mu->GetData().front(), mu_dev,
	     5000*sizeof(double), cudaMemcpyDeviceToHost);

  //tpiU->Zero();
  size_t l  = 0;
  size_t pj = 0;
  for(size_t i=0; i<5000; i+=10){
    for(size_t j=0; j<10; j++){
      pj = permutationt[i+j];
      for(size_t k=0; k<10; (k++, l++))
	tpiU->SetVal(pj, permutationt[i+k], tpiU_host[l]);
    }
  }
  double tmp0 = 0.0, tmp1 = 0.0;
  for(size_t i=5000; i<tpiU->GetNRows(); i++){
    pj = permutation[i];
    tmp0 = 1 / (psiinvun->GetVal(i, i)/tau + ub->GetVal(i));
    tmp1 = -psiinvun->GetVal(i, i) / tau;
    tpiU->SetVal(pj, pj,  tmp0);
    b->SetVal(i, b->GetVal(i) + tmp0*(delta->GetVal(pj)-ub->GetVal(i)-tmp1));
    mu->SetVal(i, tmp1*b->GetVal(i));
  }
  b->PermuteRows(permutation);
  mu->PermuteRows(permutation);

  for(size_t i=0; i<nind; i++){
    sigma2->SetVal(i, tau/psiinv->GetVal(i, i));
    mu->SetVal(i, mu->GetVal(i) * tau / psiinv->GetVal(i, i));
  }
  (*mu) += (*b);
  // update ub and w
  for(size_t i=0; i<nind; i++){
    ub->SetVal(i, clambda->GetVal(i) * ebx->GetVal(i) * exp(b->GetVal(i)));
    if(b_partial_laplace){
      double s0 = 0.0, s1 = 0.0;
      for(size_t j=0; j<gauss_hermite.GetOrder(); j++){
	double arg = sqrt(2*sigma2->GetVal(i)) * gauss_hermite.GetRoot(j);
	arg += mu->GetVal(i);
	double m = delta->GetVal(i) * arg;
	m = exp(m - clambda->GetVal(i) * ebx->GetVal(i) * exp(arg));
	s0 += exp(arg) * m * gauss_hermite.GetWeight(j);
	s1 += m * gauss_hermite.GetWeight(j);
      }
      w->SetVal(i, s0 == 0.0 ? 0.0 : s0 / s1);
    }
    else w->SetVal(i, exp(b->GetVal(i) - 0.5*tpiU->GetVal(i, i)));
  }
  auto t1 = high_resolution_clock::now();
  titer_us["Updateb"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GRMModel::UpdateBeta(){
  auto t0 = high_resolution_clock::now();
  betaprev->SetData(beta->GetData());
  Ubeta->Zero();
  Vbeta->Zero();
  // compute the sums for U and V, taking advantage of the time ordering
  size_t j = 0;
  double tmp = 0.0, tmpval = 0.0;
  PMatrix<double>* tmprow = new PMatrix<double>("tmprow", *Ubeta);
  PMatrix<double>* tmpmat = new PMatrix<double>("tmpmat", *Vbeta);
  for(size_t i=0; i<T->GetNData(); ){
    while(i == j || (i > 0 ? T->GetVal(i-1) == T->GetVal(i) : true)){
      tmp = ebx->GetVal(i) * w->GetVal(i);
      tmpval += tmp;
      tmprow->AddMultRow(0, i, tmp, *X);
      X->OuterRowProduct(i, i, tmp, tmpmat);
      i ++;
      if(i == T->GetNData()) break;
    }
    for(; j<i; j++){
      if(delta->GetVal(j) == 0.0) continue;
      Ubeta->AddMultRow(0, j, 1.0, *X);
      Ubeta->AddMultRow(0, 0, -1/tmpval, *tmprow);
      Vbeta->axpy(1/tmpval, *tmpmat);
      tmprow->OuterRowProduct(0, 0, -1/tmpval/tmpval, Vbeta);
    }
  }
  if(tmpmat) delete tmpmat;
  tmpmat = Vbeta->Inverse();
  Ubeta->RightMult(*tmpmat, tmprow);
  (*beta) += (*tmprow);
  for(size_t i=0; i<ebx->GetNData(); i++)
    ebx->SetVal(i, exp(X->InnerRowProduct(i, 0, *beta)));
  delete tmprow;
  delete tmpmat;
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateBeta"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GRMModel::UpdateLambda(){
  auto t0 = high_resolution_clock::now();
  lambdaprev->SetData(lambda->GetData());
  lambda->Zero();
  clambda->Zero();
  // update lambda and clambda with new beta and w
  size_t j = 0;
  double tmp = 0.0;
  for(size_t i=0; i<T->GetNData(); ){
    while(i == j || (i > 0 ? T->GetVal(i-1) == T->GetVal(i) : true)){
      tmp += ebx->GetVal(i) * w->GetVal(i);
      i ++;
      if(i == T->GetNData()) break;
    }
    for(; j<i; j++) lambda->SetVal(j, delta->GetVal(j)/tmp);
  }
  j = T->GetNData() - 1;
  tmp = 0.0;
  for(size_t i=T->GetNData()-1; (i>=0&&i<T->GetNData())||i==(size_t)(-1); ){
    while(i == j || (i<T->GetNData()-1 ? T->GetVal(i+1)==T->GetVal(i) : true)){
      tmp += lambda->GetVal(i);
      i --;
      if(i > T->GetNData()) break;
    }
    for(; i>T->GetNData()?j>=0:j>i; j--){
      clambda->SetVal(j, tmp);
      if(j == 0) break;
    }
  }
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateLambda"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GRMModel::UpdateTau(){
  auto t0 = high_resolution_clock::now();
  tauprev = tau;
  // compute Ub + (tau*psi)^-1 and invert in the buffer
  for(size_t i=0; i<psiinv->GetNRows(); i++)
    for(size_t j=0; j<psiinv->GetNCols(); j++){
      buffmat->SetVal(i, j, psiinv->GetVal(i, j) / tau);
      if(i == j) buffmat->SetVal(i, j, buffmat->GetVal(i, j) + ub->GetVal(i));
    }
  buffmat->BlockwiseInvert(buffmat->GetNRows()/64);
  // update tau, avoid full matrix product since taking trace
  tau = psiinv->TraceProduct(*buffmat);
  for(size_t i=0; i<b->GetNData(); i++)
    tau += b->GetVal(i) * psiinv->InnerProduct(i, 0, *b);
  tau /= nind;
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateTau"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


void GRMModel::UpdateTauGH2(){
  auto t0 = high_resolution_clock::now();
  delta->PermuteRows(permutationt);
  clambda->PermuteRows(permutationt);
  X->PermuteRows(permutationt);
  ebx->PermuteRows(permutationt);
  PMatrix<double>* a2        = new PMatrix<double>("a2",        1, 2, false);
  PMatrix<double>* b2        = new PMatrix<double>("b2",        2, 1, false);
  PMatrix<double>* delta2    = new PMatrix<double>("delta2",    1, 2, false);
  PMatrix<double>* sqrttpsi  = new PMatrix<double>("sqrttpsi",  2, 2, false);
  PMatrix<double>* sqrttpsib = new PMatrix<double>("sqrttpsib", 2, 1, false);
  PMatrix<double>* Ebb       = new PMatrix<double>("Ebb",       2, 2, false);
  double c = sqrt(0.5 * tau / (1 + sqrt(1-0.5*0.5)));
  sqrttpsi->SetVal(0, 0, c * (1+sqrt(1-0.5*0.5)));
  sqrttpsi->SetVal(0, 1, c * 0.5);
  sqrttpsi->SetVal(1, 0, sqrttpsi->GetVal(0, 1));
  sqrttpsi->SetVal(1, 1, sqrttpsi->GetVal(0, 0));
  tauprev = tau;
  tau = 0.0;
  size_t nsample = 0;
  for(size_t q=0; q<psi->GetNRows(); q+=3){
    nsample += 2;
    delta2->SetVal(0, delta->GetVal(q));
    delta2->SetVal(1, delta->GetVal(q+1));
    a2->SetVal(0, clambda->GetVal(q)   * ebx->GetVal(q));
    a2->SetVal(1, clambda->GetVal(q+1) * ebx->GetVal(q+1));
    double s0 = 0.0;
    Ebb->Zero();
    for(size_t i=0; i<gauss_hermite.GetOrder(); i++)
      for(size_t j=0; j<gauss_hermite.GetOrder(); j++){
	b2->SetVal(0, M_SQRT2 * gauss_hermite.GetRoot(i));
	b2->SetVal(1, M_SQRT2 * gauss_hermite.GetRoot(j));
	sqrttpsi->RightMult(*b2, sqrttpsib);
	double f0 = delta2->InnerProduct(0, 0, *sqrttpsib);
	f0 -= a2->GetVal(0) * exp(sqrttpsib->GetVal(0));
	f0 -= a2->GetVal(1) * exp(sqrttpsib->GetVal(1));
	f0 = exp(f0) * gauss_hermite.GetWeightProd(i, j);
	s0 += f0;
	Ebb->SetVal(0, 0, Ebb->GetVal(0, 0) +
		    f0 * sqrttpsib->GetVal(0) * sqrttpsib->GetVal(0));
	Ebb->SetVal(1, 1, Ebb->GetVal(1, 1) +
		    f0 * sqrttpsib->GetVal(1) * sqrttpsib->GetVal(1));
	Ebb->SetVal(0, 1, Ebb->GetVal(0, 1) +
		    f0 * sqrttpsib->GetVal(0) * sqrttpsib->GetVal(1));
	Ebb->SetVal(1,0, Ebb->GetVal(0,1));
      }
    (*Ebb) *= 1 / s0;
    tau += Ebb->GetVal(0, 0) - 0.5*Ebb->GetVal(0, 1);
    tau += Ebb->GetVal(1, 1) - 0.5*Ebb->GetVal(1, 0);
  }
  tau /= nsample * (1 - 0.5*0.5);
  delete a2;
  delete b2;
  delete delta2;
  delete sqrttpsi;
  delete sqrttpsib;
  delete Ebb;
  delta->PermuteRows(permutation);
  clambda->PermuteRows(permutation);
  X->PermuteRows(permutation);
  ebx->PermuteRows(permutation);
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateTauGH2"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;  
}


void GRMModel::UpdateTauGH2(vector<size_t>& bsize,
			    vector<pair<size_t, size_t> >& sample){
  auto t0 = high_resolution_clock::now();
  assert(bsize.size() == sample.size());

  /*ROOT::Math::GSLRandomEngine gsl_rand;
  gsl_rand.Initialize();
  for(size_t i=0; i<sample.size(); i++){
    if(bsize[i] < 2) continue;
    sample[i].first  = (size_t) (gsl_rand() *  bsize[i]);
    sample[i].second = (size_t) (gsl_rand() * (bsize[i]-1));
    if(sample[i].second == sample[i].first) sample[i].second = bsize[i]-1;
    }*/
  
  PMatrix<double>* a2        = new PMatrix<double>("a2",        1, 2, false);
  PMatrix<double>* b2        = new PMatrix<double>("b2",        2, 1, false);
  PMatrix<double>* delta2    = new PMatrix<double>("delta2",    1, 2, false);
  PMatrix<double>* psiinv2   = new PMatrix<double>("psiinv2",   2, 2, false);
  PMatrix<double>* sqrttpsi  = new PMatrix<double>("sqrttpsi",  2, 2, false);
  PMatrix<double>* sqrttpsib = new PMatrix<double>("sqrttpsib", 2, 1, false);
  PMatrix<double>* Ebb       = new PMatrix<double>("Ebb",       2, 2, false);
  a2->SetUseCblas(false);
  b2->SetUseCblas(false);
  delta2->SetUseCblas(false);
  //psiinv2->SetUseCblas(false);
  sqrttpsi->SetUseCblas(false);
  sqrttpsib->SetUseCblas(false);
  Ebb->SetUseCblas(false);
  tauprev = tau;
  tau = 0.0;
  size_t nsample = 0;
  size_t offset  = 0;
  for(size_t q=0; q<bsize.size(); q++){
    if(bsize[q] < 2){
      offset += bsize[q];
      continue;
    }
    nsample += 2;
    size_t qi = distance(permutation.begin(),
			 find(permutation.begin(), permutation.end(),
			      offset + sample[q].first));
    size_t qj = distance(permutation.begin(),
			 find(permutation.begin(), permutation.end(),
			      offset + sample[q].second));
    assert(qi < psi->GetNRows() && qj < psi->GetNRows());
    psiinv2->SetVal(0, 0, psi->GetVal(qi, qi));
    psiinv2->SetVal(0, 1, psi->GetVal(qi, qj));
    psiinv2->SetVal(1, 0, psi->GetVal(qj, qi));
    psiinv2->SetVal(1, 1, psi->GetVal(qj, qj));
    double s = sqrt(psi->GetVal(qi, qi)*psi->GetVal(qj, qj) -
		    psi->GetVal(qi, qj)*psi->GetVal(qi, qj));
    sqrttpsi->SetData(psiinv2->GetData());
    psiinv2->Invert();
    sqrttpsi->SetVal(0, 0, sqrttpsi->GetVal(0, 0) + s);
    sqrttpsi->SetVal(1, 1, sqrttpsi->GetVal(1, 1) + s);
    (*sqrttpsi) *= sqrt(tauprev/(psi->GetVal(qi, qi)+psi->GetVal(qj, qj)+2*s));
    delta2->SetVal(0, delta->GetVal(qi));
    delta2->SetVal(1, delta->GetVal(qj));
    a2->SetVal(0, clambda->GetVal(qi) * ebx->GetVal(qi));
    a2->SetVal(1, clambda->GetVal(qj) * ebx->GetVal(qj));
    double s0 = 0.0;
    Ebb->Zero();
    for(size_t i=0; i<gauss_hermite.GetOrder(); i++)
      for(size_t j=0; j<gauss_hermite.GetOrder(); j++){
	b2->SetVal(0, gauss_hermite.GetRootSqrt2(i));
	b2->SetVal(1, gauss_hermite.GetRootSqrt2(j));
	//sqrttpsi->RightMult(*b2, sqrttpsib);
	sqrttpsib->SetVal(0, 
			  sqrttpsi->GetVal(0) * b2->GetVal(0) +
			  sqrttpsi->GetVal(1) * b2->GetVal(1));
	sqrttpsib->SetVal(1, 
			  sqrttpsi->GetVal(2) * b2->GetVal(0) +
			  sqrttpsi->GetVal(3) * b2->GetVal(1));
	double f0 = delta2->InnerProduct(0, 0, *sqrttpsib);
	f0 -= a2->GetVal(0) * exp(sqrttpsib->GetVal(0));
	f0 -= a2->GetVal(1) * exp(sqrttpsib->GetVal(1));
	f0 = exp(f0) * gauss_hermite.GetWeightProd(i, j);
	s0 += f0;
	Ebb->SetVal(0, Ebb->GetVal(0) +
		    f0 * sqrttpsib->GetVal(0) * sqrttpsib->GetVal(0));
	Ebb->SetVal(3, Ebb->GetVal(3) +
		    f0 * sqrttpsib->GetVal(1) * sqrttpsib->GetVal(1));
	Ebb->SetVal(1, Ebb->GetVal(1) +
		    f0 * sqrttpsib->GetVal(0) * sqrttpsib->GetVal(1));
	Ebb->SetVal(2, Ebb->GetVal(1));
      }
    (*Ebb) *= 1 / s0;
    tau += psiinv2->TraceProduct(*Ebb);
    offset += bsize[q];
  }
  tau /= nsample;
  delete a2;
  delete b2;
  delete delta2;
  delete sqrttpsi;
  delete sqrttpsib;
  delete Ebb;
  auto t1 = high_resolution_clock::now();
  titer_us["UpdateTauGH2"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
}


bool GRMModel::Iterateb(size_t niter){
  if(!b_from_last) b->Zero();
  for(size_t iter=0; iter<niter; iter++){
    Updateb();
    double bnorm = 0.0;
    for(size_t i=0; i<b->GetNData(); i++)
      bnorm += abs(b->GetVal(i) - bprev->GetVal(i));
    if(bnorm < epsilon){
      if(verbose) cout << "b iterations: " << iter << endl;
      return true;
    }
  }
  return false;
}


bool GRMModel::Iterate(size_t niter, size_t nbiter, bool updateTau){
  for(size_t iter=0; iter<niter; iter++){
    auto t0 = high_resolution_clock::now();
    niterations ++;
    Iterateb(nbiter);
    UpdateBeta();
    UpdateLambda();
    if(updateTau){
      UpdateTauGH2();
      taunll.push_back(tau);
      betanll.push_back(beta->GetData());
    }
    auto t1 = high_resolution_clock::now();
    titer_us["Total"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    bool converge = CheckConvergence();
    for(auto& t : titer_us) ttotal_us[t.first] += t.second;
    if(converge) return true;
  }
  return false;
}


bool GRMModel::IterateGH2(size_t niter, size_t nbiter,
			  vector<size_t>& bsize,
			  vector<pair<size_t, size_t> >& sample){
  for(size_t iter=0; iter<niter; iter++){
    auto t0 = high_resolution_clock::now();
    niterations ++;
    Iterateb(nbiter);
    auto t2 = high_resolution_clock::now();
    UpdateBeta();
    auto t3 = high_resolution_clock::now();
    UpdateLambda();
    auto t4 = high_resolution_clock::now();
    double tautmp = tau;
    double taumean = 0.0;
    if(sample.size() == bsize.size()){
      tau = tautmp;
      UpdateTauGH2(bsize, sample);
      taumean += tau;
    }
    else{
      vector<pair<size_t, size_t> > lsample(bsize.size(), make_pair(0, 0));
      const size_t n = bsize[0];
      for(size_t i=0; i<n; i++)
	for(size_t j=i+1; j<n; j++){
	  for(size_t k=0; k<bsize.size(); k++){
	    lsample[k].first  = i;
	    lsample[k].second = j;
	  }
	  tau = tautmp;
	  UpdateTauGH2(bsize, lsample);
	  taumean += tau;
	}
    }
    auto t5 = high_resolution_clock::now();

    cout << duration_cast<nanoseconds>(t2-t0).count()*1.0e-9 << "  "
    	 << duration_cast<nanoseconds>(t3-t2).count()*1.0e-9 << "  "
    	 << duration_cast<nanoseconds>(t4-t3).count()*1.0e-9 << "  "
    	 << duration_cast<nanoseconds>(t5-t4).count()*1.0e-9 << endl;
    
    tau = taumean;
    if(sample.size() != bsize.size()) tau /= (bsize[0] * (bsize[0]-1)) / 2;
    taunll.push_back(tau);
    betanll.push_back(beta->GetData());
    auto t1 = high_resolution_clock::now();
    titer_us["Total"] = duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    bool converge = CheckConvergence();
    for(auto& t : titer_us) ttotal_us[t.first] += t.second;
    if(converge) return true;
  }
  return false;
}


bool GRMModel::CheckConvergence(){
  auto t0 = high_resolution_clock::now();
  double bnorm = 0.0;
  for(size_t i=0; i<npar; i++)
    bnorm += abs(beta->GetVal(i) - betaprev->GetVal(i));
  double lnorm = 0.0;
  for(size_t i=0; i<nind; i++)
    lnorm = std::max(lnorm, abs(lambda->GetVal(i) - lambdaprev->GetVal(i)));
  double tnorm = abs(tau - tauprev);
  if(verbose){
    cout << scientific << "iteration " << niterations << " "
	 << setprecision(3) << titer_us["Total"] << " us:" << endl;
    beta->Print(beta->GetNData(), 4, "    ");
    cout << "    tau = " << tau << endl << "    "
	 << "||beta|| = "   << setprecision(3)   << bnorm << ", "
	 << "||lambda|| = " << setprecision(3)   << lnorm << ", "
	 << "|tau| = "      << setprecision(3)   << tnorm << endl;
  }
  auto t1 = high_resolution_clock::now();
  titer_us["CheckConvergence"]=duration_cast<nanoseconds>(t1-t0).count()*1.e-3;
  if(std::max(tnorm, std::max(bnorm, lnorm)) < epsilon) return true;
  return false;
}


void GRMModel::Solve(size_t maxiter, size_t nbiter){
  SetSeeds();
  while(!Iterate(1, nbiter) &&
	(maxiter > 0 ? niterations < maxiter : true)){ }
  // print times and solution
}


void GRMModel::SolveGH2(size_t maxiter, size_t nbiter,
		     vector<size_t>& bsize,
		     vector<pair<size_t, size_t> >& sample){
  SetSeeds();
  while(!IterateGH2(1, nbiter, bsize, sample) &&
	(maxiter > 0 ? niterations < maxiter : true)){ }
  cout << "SolveGH2: " << niterations << ", tau = " << tau << endl;
  beta->SetData(betanll.back());
  beta->Print();
  T->PermuteRows(permutationt);
  X->PermuteRows(permutationt);
  delta->PermuteRows(permutationt);
  psi->Permute(permutationt);
  psiinv->Permute(permutationt);
}


void GRMModel::GridSearchTau(size_t maxiter, size_t maxbiter,
			     double tmin, double tmax, size_t ntau){
  SetSeeds();
  taumin = tmin;
  taumax = tmax;
  taunll.resize(ntau);
  betanll.resize(ntau, vector<double>(beta->GetNData()));
  double taustep = (taumax - taumin) / taunll.size();
  vector<double> vgrad(ntau);
  for(size_t i=0; i<taunll.size(); i++){
    tau = taustep * (i + 0.5) + taumin;
    tauprev = tau;
    niterations = 0;
    while(!Iterate(1, maxbiter, false) &&
	  (maxiter > 0 ? niterations < maxiter : true)){ }
    Updateb();
    UpdateNLL();
    //UpdateTauGradient();
    //vgrad[i] = grad;
    taunll[i] = nll;
    copy(beta->GetDataConst().begin(), beta->GetDataConst().end(),
	 betanll[i].begin());
  }
  auto minel = min_element(taunll.begin(), taunll.end());
  size_t index = distance(taunll.begin(), minel);
  tau = taustep * (index + 0.5) + taumin;
  cout << "Grid search, tau = " << tau << " at index "
       << index << " of " << ntau << ", nll = " << *minel << endl;
  beta->SetData(betanll[index]);
  beta->Print();
  //for(size_t i=0; i<ntau; i++)
  //  cout << vgrad[i] << "," << endl;
  //cout << endl;
}


void GRMModel::GradientDescentTau(size_t maxgiter,
				  size_t maxiter, size_t maxbiter,
				  double tauseed, double stepseed){
  SetSeeds();
  double taustep = stepseed;
  size_t gditer = 0;
  tau      = tauseed;
  tauprev  = tauseed;
  nll      = 0.0;
  grad     = 0.0;
  Updateb();
  UpdateNLL();
  UpdateTauGradient();
  tauprev = tau;
  cout << grad << endl;
  tau -= taustep * (grad < 0 ? -1 : 1);
  while(gditer == 0 || (abs((nll-nllprev)/nllprev) > epsilon &&
			(maxgiter > 0 ? gditer < maxgiter : true))){
    gditer ++;
    niterations = 0;
    double tmp = tauprev;
    tauprev = tau;
    while(!Iterate(1, maxbiter, false) &&
	  (maxiter > 0 ? niterations < maxiter : true)){ }
    tauprev = tmp;
    Updateb();
    UpdateNLL();
    UpdateTauGradient();
    cout << tau << " " << tauprev << " " << grad << " " << gradprev << " " << (tau-tauprev)/(grad-gradprev) << endl;
    taustep = abs((tau-tauprev) / (grad-gradprev));
    tauprev = tau;
    while(tau - taustep * grad < 0.0) taustep /= 2;
    tau -= taustep * grad;
    cout << tau << endl;
    cout << "Grad descent step " << gditer << ":" << endl
	 << "tau = " << tau << ", nllprev = " << nllprev
	 << ", nll = " << nll << ", |nll/nllprev-1| = "
	 << abs((nll-nllprev)/nllprev) << endl;
  }
}


void GRMModel::GridSearchTauGrad(size_t maxiter, size_t maxbiter,
				 double tmin, double tmax, size_t ntau){
  SetSeeds();
  taumin = tmin;
  taumax = tmax;
  taunll.resize(ntau);
  betanll.resize(ntau, vector<double>(beta->GetNData()));
  double taustep = (taumax - taumin) / taunll.size();
  for(size_t i=0; i<taunll.size(); i++){
    tau = taustep * (i + 0.5) + taumin;
    tauprev = tau;
    niterations = 0;
    while(!Iterate(1, maxbiter, false) &&
	  (maxiter > 0 ? niterations < maxiter : true)){ }
    Updateb();
    UpdateTauGradient2();
    taunll[i] = grad;
    copy(beta->GetDataConst().begin(), beta->GetDataConst().end(),
	 betanll[i].begin());
    cout << endl << tau << " " << nll << endl;
  }
  size_t index = 0;
  double minabs = 1.0e18;
  for(size_t i=0; i<taunll.size(); i++)
    if(abs(taunll[i]) < minabs){
      minabs = abs(taunll[i]);
      index = i;
    }
  tau = taustep * (index + 0.5) + taumin;
  cout << "Grid search, tau = " << tau << " at index "
       << index << " of " << ntau << ", |grad| = " << minabs << endl;
  beta->SetData(betanll[index]);
  beta->Print();
}

double GRMModel::EvalNLL(double utau, vector<double>& ubeta){
  assert(beta->GetNData() == ubeta.size());
  tauprev = tau;
  betaprev->SetData(beta->GetData());
  tau = utau;
  beta->SetData(ubeta);
  Updateb();
  UpdateLambda();
  UpdateNLL();
  return nll;
}

 

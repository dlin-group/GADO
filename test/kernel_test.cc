#include "gha_kernels.h"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

int main(){

  // constants to define data size and test tolerance
  const size_t nrows   = 4;
  const size_t nblocks = 32768;
  const size_t ndata   = nrows*nrows*nblocks;
  const size_t nvec    = nrows*nblocks;
  const size_t order   = 16;
  const size_t npar    = 8;
  const double tol     = 1.0e-9;
  const double alpha   = 0.5;
  double val = 0.0;

  // host and device matrix and vector setup
  double* m0_host = (double*) malloc(sizeof(double)*ndata);
  double* m1_host = (double*) malloc(sizeof(double)*ndata);
  double* m2_host = (double*) malloc(sizeof(double)*ndata);
  double* v0_host = (double*) malloc(sizeof(double)*nvec);
  double* v1_host = (double*) malloc(sizeof(double)*nvec);
  double* v2_host = (double*) malloc(sizeof(double)*nvec);
  double* v3_host = (double*) malloc(sizeof(double)*nblocks);
  double* c0_host = (double*) malloc(sizeof(double)*order);
  size_t* c1_host = (size_t*) malloc(sizeof(size_t)*(nrows+1));
  double* p0_host = (double*) malloc(sizeof(double)*nvec*npar);
  double* p1_host = (double*) malloc(sizeof(double)*npar);
  for(size_t i=0; i<nblocks; i++){
    size_t offset = i*nrows*nrows;
    v3_host[i] = 0.0;
    for(size_t j=0; j<nrows; j++){
      size_t m = i*nrows + j;
      v0_host[m] = 1.0;
      v1_host[m] = 1.0 / (j+1);
      v2_host[m] = 0.0;
      for(size_t k=0; k<nrows; k++){
	size_t n = offset + j*nrows + k;
	if(j == k){
	  m0_host[n] = 1.0;
	  m1_host[n] = 1.0;
	}
	else{
	  m0_host[n] = 0.5;
	  m1_host[n] = 1.0 / (k+1);
	}
	m2_host[n] = 0.0;
      }
    }
  }
  for(size_t i=0; i<nrows; i++){
    c0_host[i] = 1.0 / (i+1);
    c1_host[i] = 1;
  }
  c1_host[nrows] = 1;
  for(size_t i=nrows-1; i<(size_t)-1; i--) c1_host[i] = c1_host[i+1] * order;
  for(size_t i=0; i<nvec; i++)
    for(size_t j=0; j<npar; j++){
      if(i == 0) p1_host[j] = 1.0 / (j+1);
      p0_host[i*npar+j] = 0.25 + 0.75 / (j+1) * i;
    }
  double* m0_dev = NULL, *m1_dev = NULL, *m2_dev = NULL;
  double* v0_dev = NULL, *v1_dev = NULL, *v2_dev = NULL, *v3_dev = NULL;
  double* c0_dev = NULL;
  size_t* c1_dev = NULL;
  double* p0_dev = NULL, *p1_dev = NULL;
  cudaMalloc((void**) &m0_dev, sizeof(double)*ndata);
  cudaMalloc((void**) &m1_dev, sizeof(double)*ndata);
  cudaMalloc((void**) &m2_dev, sizeof(double)*ndata);
  cudaMalloc((void**) &v0_dev, sizeof(double)*nvec);
  cudaMalloc((void**) &v1_dev, sizeof(double)*nvec);
  cudaMalloc((void**) &v2_dev, sizeof(double)*nvec);
  cudaMalloc((void**) &v3_dev, sizeof(double)*nblocks);
  cudaMalloc((void**) &c0_dev, sizeof(double)*order);
  cudaMalloc((void**) &c1_dev, sizeof(size_t)*(nrows+1));
  cudaMalloc((void**) &p0_dev, sizeof(double)*nvec*npar);
  cudaMalloc((void**) &p1_dev, sizeof(double)*npar);
  cudaMemcpy(m0_dev, m0_host,  sizeof(double)*ndata,    cudaMemcpyHostToDevice);
  cudaMemcpy(m1_dev, m1_host,  sizeof(double)*ndata,    cudaMemcpyHostToDevice);
  cudaMemcpy(m2_dev, m2_host,  sizeof(double)*ndata,    cudaMemcpyHostToDevice);
  cudaMemcpy(v0_dev, v0_host,  sizeof(double)*nvec,     cudaMemcpyHostToDevice);
  cudaMemcpy(v1_dev, v1_host,  sizeof(double)*nvec,     cudaMemcpyHostToDevice);
  cudaMemcpy(v2_dev, v2_host,  sizeof(double)*nvec,     cudaMemcpyHostToDevice);
  cudaMemcpy(v3_dev, v3_host,  sizeof(double)*nblocks,  cudaMemcpyHostToDevice);
  cudaMemcpy(c0_dev, c0_host,  sizeof(double)*order,    cudaMemcpyHostToDevice);
  cudaMemcpy(c1_dev, c1_host,  sizeof(size_t)*(nrows+1),cudaMemcpyHostToDevice);
  cudaMemcpy(p0_dev, p0_host,  sizeof(double)*nvec*npar,cudaMemcpyHostToDevice);
  cudaMemcpy(p1_dev, p1_host,  sizeof(double)*npar,     cudaMemcpyHostToDevice);
  
  // test the component-wise product kernel
  cout << "ComponentProduct...";
  auto t0 = high_resolution_clock::now();
  ComponentProduct(v0_dev, v1_dev, v2_dev, nvec);
  auto t1 = high_resolution_clock::now();
  cudaMemcpy(v2_host, v2_dev, sizeof(double)*nvec, cudaMemcpyDeviceToHost);
  bool pass = true;
  auto t2 = high_resolution_clock::now();
  for(size_t i=0; i<nvec; i++){
    val = v0_host[i] * v1_host[i];
    if(abs(val-v2_host[i]) > tol){
      if(pass) cout << endl;
      if(i < 32)
	cout << i << " " << v0_host[i] << " " << v1_host[i] << " "
	     << v2_host[i] << " " << val << endl;
      pass = false;
    }
  }
  auto t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(v2_host, 0.0, nvec);
  cudaMemset(v2_dev, 0, sizeof(double)*nvec);

  // test the component-wise exponential kernel
  cout << "ComponentExponential...";
  t0 = high_resolution_clock::now();
  ComponentExponential(v1_dev, v2_dev, nvec);
  t1 = high_resolution_clock::now();
  cudaMemcpy(v2_host, v2_dev, sizeof(double)*nvec, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<nvec; i++){
    val = exp(v1_host[i]);
    if(abs(val-v2_host[i]) > tol){
      if(pass) cout << endl;
      if(i < 32)
	cout << i << " " << v1_host[i] << " " << v2_host[i] << " " << val<<endl;
      pass = false;
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(v2_host, 0.0, nvec);
  cudaMemset(v2_dev, 0, sizeof(double)*nvec);
  
  // test the component-wise difference exponential kernel
  cout << "DifferenceExponential...";
  cudaMemcpy(v2_dev, v0_dev, sizeof(double)*nvec, cudaMemcpyDeviceToDevice);
  t0 = high_resolution_clock::now();
  DifferenceExponential(v2_dev, v1_dev, alpha, nvec);
  t1 = high_resolution_clock::now();
  cudaMemcpy(v2_host, v2_dev, sizeof(double)*nvec, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<nvec; i++){
    val = alpha * exp(v0_host[i] - v1_host[i]);
    if(abs(val-v2_host[i]) > tol){
      if(pass) cout << endl;
      if(i < 32)
	cout << i << " " << v0_host[i] << " " << v1_host[i] << " "
	     << v2_host[i] << " " << val << endl;
      pass = false;
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(v2_host, 0.0, nvec);
  cudaMemset(v2_dev, 0, sizeof(double)*nvec);

  // test the divide modulus kernel
  cout << "DivideMod...";
  cudaMemcpy(m2_dev, m0_dev, sizeof(double)*ndata, cudaMemcpyDeviceToDevice);
  t0 = high_resolution_clock::now();
  DivideMod(m2_dev, v1_dev, nrows, ndata);
  t1 = high_resolution_clock::now();
  cudaMemcpy(m2_host, m2_dev, sizeof(double)*ndata, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<ndata; i++){
    val = m0_host[i] / v1_host[i/nrows];
    if(abs(val-m2_host[i]) > tol){
      if(pass) cout << endl;
      if(i < 32)
	cout << i << " " << m0_host[i] << " " << v1_host[i/nrows] << " "
	     << m2_host[i] << " " << val << endl;
      pass = false;
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(m2_host, 0.0, ndata);
  cudaMemset(m2_dev, 0, sizeof(double)*ndata);

  // test the exponential modulus product kernel
  cout << "ExpModProduct...";
  cudaMemcpy(m2_dev, m0_dev, sizeof(double)*ndata, cudaMemcpyDeviceToDevice);
  t0 = high_resolution_clock::now();
  ExpModProduct(m2_dev, m1_dev, v1_dev, nrows, ndata);
  t1 = high_resolution_clock::now();
  cudaMemcpy(m2_host, m2_dev, sizeof(double)*ndata, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<ndata; i++){
    val = m0_host[i] + exp(m1_host[i]) * v1_host[i/nrows];
    if(abs(val-m2_host[i]) > tol){
      if(pass) cout << endl;
      if(i < 32)
	cout << i << " " << m0_host[i] << " " << m1_host[i] << " "
	     << v1_host[i/nrows] << " " << m2_host[i] << " " << val << endl;
      pass = false;
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(m2_host, 0.0, ndata);
  cudaMemset(m2_dev, 0, sizeof(double)*ndata);

  // test the cyclic permutation kernel
  if(nrows <= 4){
    cout << "CyclicPerm...";
    double ctsum = 0.0, gtsum = 0.0;
    pass = true;
    for(size_t i=0; i<c1_host[0]; i++){
      t0 = high_resolution_clock::now();
      CyclicPerm(c0_dev, c1_dev, v2_dev, i, nrows, nvec);
      t1 = high_resolution_clock::now();
      cudaMemcpy(v2_host, v2_dev, sizeof(double)*nvec, cudaMemcpyDeviceToHost);
      t2 = high_resolution_clock::now();
      for(size_t j=0; j<nblocks; j++){
	for(size_t k=0; k<nrows; k++){
	  val = c0_host[(i%c1_host[k])/c1_host[k+1]];
	  if(abs(val-v2_host[j*nrows+k]) > tol){
	    if(pass) cout << endl;
	    //if(i < 4 && j*nrows+k < 32)
	      cout << i << " " << j << " " << k << " " << " "
		   << v2_host[j*nrows+k] << " " << val << endl;
	    pass = false;
	  }
	}
      }
      t3 = high_resolution_clock::now();
      memset(v2_host, 0.0, nvec);
      cudaMemset(v2_dev, 0, sizeof(double)*nvec);
      ctsum += duration_cast<nanoseconds>(t3-t2).count()*1.0e-3;
      gtsum += duration_cast<nanoseconds>(t1-t0).count()*1.0e-3;
    }
    if(pass) cout << "passed" << endl;
    cout << "  CPU: " << scientific << setprecision(4)
	 << ctsum << " us, GPU: " << gtsum << " us" << endl;
  }

  // test the strided inner product kernel
  cout << "InnerProdStride...";
  t0 = high_resolution_clock::now();
  InnerProdStride(v0_dev, v1_dev, v3_dev, nrows, nblocks);
  t1 = high_resolution_clock::now();
  cudaMemcpy(v3_host, v3_dev, sizeof(double)*nblocks, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<nblocks; i++){
    val = 0.0;
    for(size_t j=0; j<nrows; j++)
      val += v0_host[i*nrows+j] * v1_host[i*nrows+j];
    if(abs(val-v3_host[i]) > tol){
      if(pass) cout << endl;
      if(i < 8)
	for(size_t j=0; j<nrows; j++)
	  cout << i << " " << j << " " << v0_host[i*nrows+j] << " "
	       << v1_host[i*nrows+j] << " " << v3_host[i] << " " << val << endl;
      pass = false;
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(v3_host, 0.0, nvec);
  cudaMemset(v3_dev, 0, sizeof(double)*nblocks);

  // test the strided inner product kernel
  cout << "InnerRowProd...";
  t0 = high_resolution_clock::now();
  InnerRowProd(p0_dev, p1_dev, v2_dev, npar, nvec);
  t1 = high_resolution_clock::now();
  cudaMemcpy(v2_host, v2_dev, sizeof(double)*nvec, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<nvec; i++){
    val = 0.0;
    for(size_t j=0; j<npar; j++)
      val += p0_host[i*npar+j] * p1_host[j];
    if(abs(val-v2_host[i]) > tol){
      if(pass) cout << endl;
      //if(i < 8)
	for(size_t j=0; j<npar; j++)
	  cout << i << " " << j << " " << p0_host[i*npar+j] << " "
	       << p1_host[j] << " " << v2_host[i] << " " << val << endl;
      pass = false;
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(v2_host, 0.0, nvec);
  cudaMemset(v2_dev, 0, sizeof(double)*nvec);

  // test the outer modulus product kernel
  cout << "OuterModProd...";
  for(size_t i=0; i<nblocks; i++) v3_host[i] = 1.0 / (i+1);
  cudaMemcpy(v3_dev, v3_host, sizeof(double)*nblocks, cudaMemcpyHostToDevice);
  t0 = high_resolution_clock::now();
  OuterModProd(v0_dev, v3_dev, m2_dev, nrows, ndata);
  t1 = high_resolution_clock::now();
  cudaMemcpy(m2_host, m2_dev, sizeof(double)*ndata, cudaMemcpyDeviceToHost);
  pass = true;
  t2 = high_resolution_clock::now();
  for(size_t i=0; i<nblocks; i++){
    size_t offset = i*nrows*nrows;
    for(size_t j=0; j<nrows; j++){
      for(size_t k=0; k<nrows; k++){
	val = v0_host[i*nrows+j] * v0_host[i*nrows+k] * v3_host[i];
	if(abs(val-m2_host[offset+j*nrows+k]) > tol){
	  if(pass) cout << endl;
	  //if(i < 8)
	    cout << i << " " << j << " " << k << " " << v0_host[i*nrows+j]
		 << " " << v0_host[i*nrows+k] << " " << v3_host[i] << " " 
		 << m2_host[offset+j*nrows+k] << " " << val << endl;
	  pass = false;
	}
      }
    }
  }
  t3 = high_resolution_clock::now();
  if(pass) cout << "passed" << endl;
  cout << "  CPU: " << scientific << setprecision(4)
       << duration_cast<nanoseconds>(t3-t2).count()*1.0e-3 << " us, GPU: "
       << duration_cast<nanoseconds>(t1-t0).count()*1.0e-3 << " us" << endl;
  memset(m2_host, 0.0, ndata);
  cudaMemset(m2_dev, 0, sizeof(double)*ndata);
  memset(v3_host, 0.0, nblocks);
  cudaMemset(v3_dev, 0, sizeof(double)*nblocks);
    
  // cleanup
  free(m0_host); free(m1_host); free(m2_host);
  free(v0_host); free(v1_host); free(v2_host); free(v3_host);
  free(c0_host); free(c1_host);
  free(p0_host); free(p1_host);
  cudaFree(m0_dev); cudaFree(m1_dev); cudaFree(m2_dev);
  cudaFree(v0_dev); cudaFree(v1_dev); cudaFree(v2_dev); cudaFree(v3_dev);
  cudaFree(c0_dev); cudaFree(c1_dev);
  cudaFree(p0_dev); cudaFree(p1_dev);
  
  return 0;
}

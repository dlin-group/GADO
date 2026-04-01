#ifndef __gha_kernels__
#define __gha_kernels__

#include <cstddef>

#define GHA_THREADS_PER_BLOCK 512

using namespace std;

void CUDAReduce(double* x, double* y, unsigned int n);

void APBX(double* x, double alpha, double beta, size_t n);

void ComponentProduct(double* x, double* y, double* z, size_t n);

void ComponentExponential(double* x, double* ex, size_t n);

void DifferenceExponential(double* x, double* y, double alpha, size_t n);

void ProdMod(double* x, double* y, double* z, size_t m, size_t n);

void ProdMod2(double* x, double* y, double* z, size_t m, size_t m2, size_t n);

void DivideMod(double* x, double* y, size_t m, size_t n);

void ExpModProduct(double* x, double* y, double* z, size_t m, size_t n);

void CyclicPerm(double* x, size_t* y, double* z, size_t m, size_t d, size_t n);

void InnerProdStride(double* x, double* y, double* z, size_t m, size_t n);

void InnerRowProd(double* x, double* y, double* z, size_t m, size_t n);

void OuterModProd(double* x, double* y, double* z, size_t m, size_t n);

void TraceProd(double* x, double* y, double* z, size_t m, size_t m2, size_t n);

void SetS2(double* s2, double* ebx, double* clambda,
	   double* x,  double* t,   double* t_unq,
	   size_t n, size_t o, size_t L, size_t M, size_t N);

void ScoreTestSim(double* y, double* g, double* st,
		  double v, size_t nind, size_t n);
#endif

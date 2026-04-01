#include "gha_kernels.h"
#include <cuda_runtime.h>
#include <iostream>

// warp_reduce and reduce_kernel from:
//   https://github.com/rimikadhara67/Parallel-Reduction

template <unsigned int blockSize>
__device__ void warp_reduce(volatile double* s, unsigned int t){
  if(blockSize >= 64) s[t] += s[t + 32];
  if(blockSize >= 32) s[t] += s[t + 16];
  if(blockSize >= 16) s[t] += s[t + 8];
  if(blockSize >= 8)  s[t] += s[t + 4];
  if(blockSize >= 4)  s[t] += s[t + 2];
  if(blockSize >= 2)  s[t] += s[t + 1];
}

template <unsigned int blockSize>
__global__ void reduce_kernel(double* x, double* y, unsigned int n){
  extern __shared__ double s[];
  unsigned int t = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + t;
  unsigned int gridSize = blockSize*2*gridDim.x;
  s[t] = 0;
  while(i < n){
    s[t] += x[i] + x[i+blockSize];
    i += gridSize;
  }
  __syncthreads();
  if(blockSize >= 1024){ if(t < 512){ s[t] += s[t + 512]; } __syncthreads(); }
  if(blockSize >=  512){ if(t < 256){ s[t] += s[t + 256]; } __syncthreads(); }
  if(blockSize >=  256){ if(t < 128){ s[t] += s[t + 128]; } __syncthreads(); }
  if(blockSize >=  128){if(t < 64){ s[t] += s[t + 64]; } __syncthreads(); }
  if(t < 32) warp_reduce<blockSize>(s, t);
  if(t == 0) y[blockIdx.x] = s[0];
}

__global__ void apbx_kernel(double* x, double alpha, double beta, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = alpha + beta * x[i];
}

__global__ void component_product_kernel(double* x, double* y,
					 double* z, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) z[i] = x[i] * y[i];
}

__global__ void component_exponential_kernel(double* x, double* ex, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) ex[i] = exp(x[i]);
}

__global__ void difference_exponential_kernel(double* x, double* y,
					      double alpha, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = alpha * exp(x[i] - y[i]);
}

__global__ void prod_mod_kernel(double* x, double* y, double* z,
				size_t m, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) z[i] += x[i] * y[i/m];
}

__global__ void prod_mod2_kernel(double* x, double* y, double* z,
				 size_t m, size_t m2, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) z[i] += x[i/m] * x[(i/m2)*m+(i%m)] * y[i/m2];
}

__global__ void divide_mod_kernel(double* x, double* y, size_t m, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = x[i] / y[i/m];
}

__global__ void exp_mod_product_kernel(double* x, double* y, double* z,
				       size_t m, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = x[i] + exp(y[i]) * z[i/m];
}

__global__ void cyclic_perm_kernel(double* x, size_t* y, double* z,
				   size_t m, size_t d, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) z[i] = x[(m % y[(i%d)]) / y[(i%d)+1]];
}

__global__ void inner_prod_stride_kernel(double* x, double* y, double* z,
					 size_t m, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    z[i] = 0.0;
    size_t k = i * m;
    for(size_t j=0; j<m; (j++, k++)) z[i] += x[k] * y[k];
  }
}

__global__ void inner_row_prod_kernel(double* x, double* y, double* z,
				      size_t m, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    z[i] = 0.0;
    for(size_t j=0; j<m; j++) z[i] += x[i*m+j] * y[j];
  }
}

__global__ void outer_mod_prod_kernel(double* x, double* y, double* z,
				      size_t m, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    size_t j = (i / (m*m)) * m;
    z[i] += x[j + ((i/m)%m)] * x[j + (i%m)] * y[i/(m*m)];
  }
}

__global__ void trace_prod_kernel(double* x, double* y, double* z,
				  size_t m, size_t m2, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    z[i] = 0.0;
    size_t l = m2 * i;
    for(size_t j=0; j<m; j++)
      for(size_t k=0; k<m; k++)
	z[i] += x[l+j*m+k] * y[l+k*m+j];
  }
}

__global__ void set_s2_kernel(double* s2, double* ebx, double* clambda,
			      double* x,  double* t,   double* t_unq,
			      size_t n, size_t o, size_t L, size_t M, size_t N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    double tmp = 0.0;
    for(size_t k=0,r=o; k<n; (k++,r++)){
      tmp = ebx[r];
      for(size_t l=0; l<L; l++) s2[k*N+l]   = clambda[r] * tmp * x[r*L+l];
      for(size_t l=0; l<M; l++) s2[k*N+l+L] = (t[r] >= t_unq[l] ? tmp : 0.0);
    }
  }
}

__global__ void score_test_sim_kernel(double* y, double* g, double* st,
				      double v, size_t nind, size_t n){
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if(t < n){
    double u = 0.0;
    for(size_t i=0, j=t*nind; i<nind; (i++, j++)) u += y[j] * g[i];
    st[t] = max(st[t], u * u / v);
  }
}


void CUDAReduce(double* x, double* y, unsigned int n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid((n+2*GHA_THREADS_PER_BLOCK-1) / (2*GHA_THREADS_PER_BLOCK));
  reduce_kernel<GHA_THREADS_PER_BLOCK><<<blocksPerGrid,threadsPerBlock,GHA_THREADS_PER_BLOCK*sizeof(double)>>>(x, y, n);
}

void APBX(double* x, double alpha, double beta, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  apbx_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, alpha, beta, n);
}

void ComponentProduct(double* x, double* y, double* z, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  component_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, z, n);
}  
  
void ComponentExponential(double* x, double* ex, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  component_exponential_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, ex, n);
}

void DifferenceExponential(double* x, double* y, double alpha, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  difference_exponential_kernel<<<blocksPerGrid,threadsPerBlock>>>(x,y,alpha,n);
}

void ProdMod(double* x, double* y, double* z, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  prod_mod_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, m, n);
}

void ProdMod2(double* x, double* y, double* z, size_t l, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  prod_mod2_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, l, m, n);
}

void DivideMod(double* x, double* y, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  divide_mod_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, m, n);
}

void ExpModProduct(double* x, double* y, double* z, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  exp_mod_product_kernel<<<blocksPerGrid,threadsPerBlock>>>(x,y,z,m,n);
}

void CyclicPerm(double* x, size_t* y, double* z, size_t m, size_t d, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  cyclic_perm_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, m, d, n);
}

void InnerProdStride(double* x, double* y, double* z, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  inner_prod_stride_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, m, n);
}

void InnerRowProd(double* x, double* y, double* z, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  inner_row_prod_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, m, n);
}

void OuterModProd(double* x, double* y, double* z, size_t m, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  outer_mod_prod_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, m, n);
}

void TraceProd(double* x, double* y, double* z, size_t m, size_t m2, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  trace_prod_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, m, m2, n);
}

void SetS2(double* s2, double* ebx, double* clambda,
	   double* x,  double* t,   double* t_unq,
	   size_t n, size_t o, size_t L, size_t M, size_t N){
  dim3 threadsPerBlock = n;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  set_s2_kernel<<<blocksPerGrid,threadsPerBlock>>>(s2, ebx, clambda,
						   x, t, t_unq, n, o, L, M, N);
}

void ScoreTestSim(double* y, double* g, double* st,
		  double v, size_t nind, size_t n){
  dim3 threadsPerBlock = GHA_THREADS_PER_BLOCK;
  dim3 blocksPerGrid(ceil(n/(double)GHA_THREADS_PER_BLOCK));
  score_test_sim_kernel<<<blocksPerGrid,threadsPerBlock>>>(y, g, st, v,nind,n);
}
  
